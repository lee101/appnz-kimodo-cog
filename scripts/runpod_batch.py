"""Launch one bounded Kimodo batch, publish it to R2, and always terminate the pod."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

RUNPOD_API = "https://api.runpod.io/graphql"
TAG = "appnz-kimodo-batch"
DEFAULT_MAX_MINUTES = 45
DEFAULT_DAILY_MINUTES = 60
KIMODO_COMMIT = "1aece8c124d73d255ceff5086d983b844c9f4e94"
GPU_PREFERENCE = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA A40",
)
ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "staging" / "budget.json"


def env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def gql(query: str, variables: dict | None = None) -> dict:
    request = urllib.request.Request(
        RUNPOD_API,
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {env('RUNPOD_API_KEY')}",
            "user-agent": "appnz-kimodo-batch/1",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"])[:800])
    return result["data"]


def r2_client():
    endpoint = os.getenv("R2_ENDPOINT", "").strip()
    if not endpoint:
        endpoint = f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=env("CLOUDFLARE_R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("CLOUDFLARE_R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )


def load_budget() -> dict:
    try:
        return json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"daily_gpu_minutes_cap": DEFAULT_DAILY_MINUTES, "spent": {}}


def charge_budget(minutes: int) -> None:
    value = load_budget()
    today = dt.date.today().isoformat()
    spent = float(value.setdefault("spent", {}).get(today, 0))
    cap = float(value.get("daily_gpu_minutes_cap", DEFAULT_DAILY_MINUTES))
    if spent + minutes > cap:
        raise RuntimeError(f"daily GPU budget would exceed {cap:.0f} minutes ({spent:.0f} spent)")
    value["spent"][today] = spent + minutes
    BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_PATH.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def create_pod(name: str, image: str, command: str, ttl_epoch: int) -> str:
    mutation = """
    mutation($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) { id }
    }"""
    last_error = None
    for gpu in GPU_PREFERENCE:
        variables = {
            "input": {
                "name": name,
                "imageName": image,
                "gpuTypeId": gpu,
                "cloudType": "COMMUNITY",
                "gpuCount": 1,
                "containerDiskInGb": 50,
                "volumeInGb": 0,
                "dockerArgs": f"bash -lc {shlex.quote(command)}",
                "env": [{"key": "APPNZ_TTL_EPOCH", "value": str(ttl_epoch)}],
            }
        }
        try:
            pod = gql(mutation, variables)["podFindAndDeployOnDemand"]
            if pod and pod.get("id"):
                return pod["id"]
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(f"no capacity on preferred GPUs: {last_error}")


def terminate(pod_id: str) -> None:
    gql('mutation($id: String!) { podTerminate(input:{podId:$id}) }', {"id": pod_id})


def pod_status(pod_id: str) -> dict | None:
    query = """query($id: String!) { pod(input:{podId:$id}) {
      id desiredStatus latestTelemetry { state } runtime { uptimeInSeconds }
    } }"""
    return gql(query, {"id": pod_id}).get("pod")


def publish_archive(archive_path: Path, *, bucket: str, prefix: str, public_base: str) -> str:
    client = r2_client()
    with tempfile.TemporaryDirectory(prefix="kimodo-publish-") as temp:
        extracted = Path(temp)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (extracted / member.filename).resolve()
                if extracted.resolve() not in target.parents and target != extracted.resolve():
                    raise RuntimeError(f"unsafe archive member: {member.filename}")
            archive.extractall(extracted)
        index_path = extracted / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for asset in index["assets"]:
            relative = asset.pop("assetPath")
            source = extracted / relative
            key = f"{prefix}/{relative}"
            content_type = "application/octet-stream" if source.suffix == ".bvh" else "application/json"
            client.upload_file(
                str(source),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type, "CacheControl": "public,max-age=31536000,immutable"},
            )
            asset["assetUrl"] = f"{public_base}/{key}"
            metadata_relative = asset.get("metadataPath")
            if metadata_relative:
                metadata_source = extracted / metadata_relative
                metadata_key = f"{prefix}/{metadata_relative}"
                client.upload_file(
                    str(metadata_source),
                    bucket,
                    metadata_key,
                    ExtraArgs={"ContentType": "application/json",
                               "CacheControl": "public,max-age=31536000,immutable"},
                )
        published_index = extracted / "published-index.json"
        published_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        index_key = f"{prefix}/index.json"
        client.upload_file(
            str(published_index),
            bucket,
            index_key,
            ExtraArgs={"ContentType": "application/json", "CacheControl": "public,max-age=300"},
        )
    return f"{public_base}/{index_key}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="https://github.com/lee101/appnz-kimodo-cog.git")
    parser.add_argument("--ref", required=True, help="immutable public git commit SHA")
    parser.add_argument("--manifest", default="manifests/avatar-core.json")
    parser.add_argument("--max-minutes", type=int, default=DEFAULT_MAX_MINUTES)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--image", default="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04")
    parser.add_argument("--bucket", default=os.getenv("APPSTATIC_BUCKET", "appstatic"))
    parser.add_argument("--prefix", default="app/static/animation-library/generated/kimodo-v1")
    parser.add_argument("--public-base", default=os.getenv("CLOUDFLARE_CDN_DOMAIN", "https://appstatic.app.nz"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "kimodo-avatar-core.zip"))
    args = parser.parse_args()
    if "://" not in args.public_base:
        args.public_base = "https://" + args.public_base
    if not 5 <= args.max_minutes <= 60:
        raise SystemExit("--max-minutes must be 5-60")
    if len(args.ref) != 40 or any(character not in "0123456789abcdef" for character in args.ref.lower()):
        raise SystemExit("--ref must be an immutable commit SHA")
    if not 20 <= args.diffusion_steps <= 150 or not 1 <= args.num_samples <= 4:
        raise SystemExit("--diffusion-steps must be 20-150 and --num-samples must be 1-4")

    charge_budget(args.max_minutes)
    client = r2_client()
    staging_key = f"staging/kimodo/{int(time.time())}-{args.ref[:12]}.zip"
    log_key = staging_key.removesuffix(".zip") + ".log"
    upload_url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": args.bucket, "Key": staging_key, "ContentType": "application/zip"},
        ExpiresIn=args.max_minutes * 60 + 600,
    )
    log_upload_url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": args.bucket, "Key": log_key, "ContentType": "text/plain"},
        ExpiresIn=args.max_minutes * 60 + 600,
    )
    raw_manifest = f"https://raw.githubusercontent.com/lee101/appnz-kimodo-cog/{args.ref}/{args.manifest}"
    job_command = " && ".join(
        [
            "set -euo pipefail",
            "export HF_HOME=/workspace/huggingface",
            "job_dir=$(mktemp -d /workspace/appnz-kimodo-XXXXXX)",
            "git clone --filter=blob:none --no-checkout " + shlex.quote(args.repo) + ' "$job_dir"',
            'cd "$job_dir"',
            "git checkout --detach " + shlex.quote(args.ref),
            "apt-get update -qq",
            "apt-get install -y -qq cmake ninja-build",
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements-batch.txt",
            'kimodo_dir=$(mktemp -d /workspace/kimodo-source-XXXXXX)',
            'git clone --filter=blob:none https://github.com/nv-tlabs/kimodo.git "$kimodo_dir"',
            'git -C "$kimodo_dir" checkout --detach ' + KIMODO_COMMIT,
            'git -C "$kimodo_dir" apply "$job_dir/patches/kimodo-python3-cmake.patch"',
            'python -m pip install --no-deps "$kimodo_dir"',
            "curl -fsSL " + shlex.quote(raw_manifest) + " -o /workspace/batch.json",
            "python batch.py --manifest /workspace/batch.json --output /workspace/kimodo-motions.zip "
            f"--diffusion-steps {args.diffusion_steps} --num-samples {args.num_samples} --upload-url "
            + shlex.quote(upload_url),
        ]
    )
    command = (
        "set -o pipefail; ("
        + job_command
        + ") 2>&1 | tee /tmp/appnz-kimodo-batch.log; "
        + "status=${PIPESTATUS[0]}; "
        + "curl -fsS -X PUT -H 'content-type: text/plain' --data-binary @/tmp/appnz-kimodo-batch.log "
        + shlex.quote(log_upload_url)
        + " >/dev/null; exit $status"
    )
    ttl_epoch = int(time.time()) + args.max_minutes * 60
    pod_name = f"{TAG}-{int(time.time())}"
    pod_id = create_pod(pod_name, args.image, command, ttl_epoch)
    print(f"created {pod_name} ({pod_id}); hard cap {args.max_minutes} minutes")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    local_log = output.with_suffix(".log")
    completed = False
    try:
        deadline = time.time() + args.max_minutes * 60
        while time.time() < deadline:
            try:
                client.head_object(Bucket=args.bucket, Key=staging_key)
                client.download_file(args.bucket, staging_key, str(output))
                break
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey"}:
                    raise
            try:
                client.head_object(Bucket=args.bucket, Key=log_key)
                log = client.get_object(Bucket=args.bucket, Key=log_key)["Body"].read().decode(
                    "utf-8", errors="replace"
                )
                local_log.write_text(log, encoding="utf-8")
                raise RuntimeError("RunPod bootstrap failed:\n" + "\n".join(log.splitlines()[-80:]))
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey"}:
                    raise
            status = pod_status(pod_id)
            telemetry = (status or {}).get("latestTelemetry") or {}
            if (
                status is None
                or status.get("desiredStatus") in {"EXITED", "TERMINATED"}
                or telemetry.get("state") == "exited"
            ):
                raise RuntimeError("RunPod exited before uploading the batch artifact")
            time.sleep(20)
        else:
            raise RuntimeError("batch did not upload its artifact before the hard deadline")
        index_url = publish_archive(
            output,
            bucket=args.bucket,
            prefix=args.prefix,
            public_base=args.public_base.rstrip("/"),
        )
        completed = True
        print(json.dumps({"archive": str(output), "indexUrl": index_url}, indent=2))
    finally:
        try:
            client.delete_object(Bucket=args.bucket, Key=staging_key)
            if completed:
                client.delete_object(Bucket=args.bucket, Key=log_key)
        finally:
            terminate(pod_id)
            print(f"terminated RunPod pod {pod_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
