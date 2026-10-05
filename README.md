# app.nz Kimodo Motion Cog

An open adapter around NVIDIA Kimodo for text-to-human-motion generation. It
loads the SOMA-RP v1.1 checkpoint once, uses a public 4-bit LLM2Vec text
encoder, and exports each result as raw Kimodo NPZ plus standard-T-pose SOMA-77
BVH. The same implementation powers Cog predictions, RunPod Serverless, and a
bounded one-shot batch publisher for the app.nz animation library.

## What is included

- `predict.py`: one prompt to a downloadable ZIP for Cog/app.nz.
- `runpod_handler.py`: RunPod Serverless adapter.
- `batch.py`: many prompts through one warm model process.
- `manifests/avatar-core.json`: 35 seeded motions covering nuanced idles,
  conversation, laugh/smile/reactions, walking, running, swimming, sitting,
  object attachment, dance, celebration, and stretching.
- `scripts/runpod_batch.py`: capped RunPod pod, R2 staging, asset publication,
  generated index, and unconditional teardown.
- `docs/AUDIO2FACE_BROWSER.md`: current local/browser Audio2Face feasibility.
- `docs/QUALITY_REPORT.md`: VRM contact-sheet review and clip-selection notes.

## Model terms

The adapter code is MIT. The downloaded components are not:

- Kimodo source: Apache-2.0.
- `nvidia/Kimodo-SOMA-RP-v1.1`: NVIDIA Open Model License.
- `matbee/kimodo-llm2vec-nf4`: Meta Llama 3 Community License.

Read [THIRD_PARTY.md](THIRD_PARTY.md) before distributing an image or generated
assets. The service intentionally avoids the more restrictive SMPL-X model.

## Run as a Cog

Install [Cog](https://github.com/replicate/cog), then:

```bash
cog build -t appnz-kimodo-motion
cog predict \
  -i prompt="A relaxed character listens with tiny natural weight shifts." \
  -i duration=6 \
  -i num_samples=2
```

The first build/download is large: expect about 6 GB of model weights. An
NVIDIA GPU with at least 12 GB VRAM is recommended; the pre-quantized encoder
uses about 5 GB instead of the stock encoder's roughly 16 GB.

Kimodo source is pinned to commit
`1aece8c124d73d255ceff5086d983b844c9f4e94`. The build applies a one-line
CMake variable correction so its native post-processor links against the
container's active Python rather than Ubuntu's system Python.

## Run on app.nz

Build and push the Linux AMD64 image, then use Cog Studio:

```bash
docker build --platform linux/amd64 -t YOUR_REGISTRY/appnz-kimodo-motion:COMMIT .
docker push YOUR_REGISTRY/appnz-kimodo-motion:COMMIT

app cogs deploy kimodo-motion
app cogs run kimodo-motion --input '{
  "prompt": "A friendly character gives a small wave and settles naturally.",
  "duration": 4,
  "seed": 304
}'
```

The declarative UI is in `appnz.schema.json`; runtime metadata and the
scale-to-zero recommendation are in `appnz.cog.json`. Pin deployments to an
immutable image digest rather than `latest`.

## Generate and publish the core batch on RunPod

The launcher accepts only an immutable public git SHA, reserves at most 60
GPU-minutes per day locally, never sends R2 credentials into the pod, and
terminates the pod in `finally`. It stages the result through an expiring
presigned URL, publishes the BVH/metadata files, and prints the index URL.

```bash
set -a
. ../app-site/.env
set +a

python -m pip install boto3
python scripts/runpod_batch.py --ref "$(git rev-parse HEAD)" --max-minutes 45
```

Before and after a run, verify there are no tagged live pods:

```bash
python scripts/runpod_status.py
```

Configure app.nz with the printed public URL:

```bash
ANIMATION_GENERATED_INDEX_URL=https://appstatic.app.nz/app/static/animation-library/generated/kimodo-v1/index.json
```

Generated library assets point directly to immutable BVH files and include the
prompt, tags, model/license provenance, and metadata path. NPZ files remain in
the retrieved local archive for quality analysis and future retargeting.

## Quality review

Text-to-motion output should be treated as candidates, not automatically
shipped animation. Review at least:

- the first, middle, and last second of every clip;
- foot sliding, ground penetration, discontinuities, and root drift;
- hand/body collision and correct left/right actions;
- loop seam quality for items marked `loop`;
- retargeting on the actual VRM skeleton at normal and slow speed.

Keep the best seed, revise ambiguous prompts, and regenerate only failures.
The raw NPZ is retained so quality metrics and future retargeters are not
limited by BVH conversion.

## Tests

Contract tests do not download weights:

```bash
python -m pip install -e '.[test]'
pytest
ruff check .
```

## Reusable animation API (CPU only)

`library_server.py` serves the existing generated archive, never loads model
weights, and never provisions a GPU. This is separate from the Kimodo Cog
fresh-generation path and must not be labeled as NVIDIA ACE generation.

```bash
python library_server.py --archive outputs/kimodo-avatar-core.zip --port 9092
curl 'http://127.0.0.1:9092/v1/animations/library?q=wave'
curl http://127.0.0.1:9092/v1/animations/generations \
  -H 'Content-Type: application/json' \
  -d '{"provider":"kimodo-library","asset_id":"kimodo-wave-friendly","rig_type":"biped"}'
python library_server.py --export ../animflow/public
```

The library includes 35 generated candidates with model, seed, prompt, license,
SHA-256, and immutable content-addressed BVH URLs. Reuse returns a stable result
ID, `cached: true`, and zero new GPU-generation seconds. New prompts/providers
are rejected rather than silently substituted with existing or procedural data.
`GET /health` reports that GPU loading and new generation are disabled.
`deploy/kimodo-library.service` bounds this loopback-only API to 256 MiB RAM,
zero swap, one CPU, and 16 tasks; deploy a pinned release under its working path.
Put authentication/rate limits in front before exposing a fresh-generation API.

AnimFlow's Motion Capture sidebar browses these candidates lazily, verifies BVH
checksums, and bakes SOMA-77 rotations to portable `vrm-pose-clip/v1`. Metadata
and provenance survive project indexing and JSON export. Wolf/quadruped claims
are rejected. Loops stay off: a requested loop is not evidence of a reviewed
seam. The browser preview folds unmapped ancestors, omits finger/facial and
horizontal root motion, normalizes vertical motion against the first frame,
and scales root height to the target avatar's leg length. Review target-avatar
contacts; this is not a universal anatomical retargeter.

For fresh combat candidates, `manifests/combat-candidates.json` contains three
short one-shot biped attacks/reactions (11 seconds total, one sample each).
Use the existing bounded batch launcher only with an approved spend cap. Start
with a 60-step candidate pass, then rerun only failed seeds at 100 steps; lower
step counts are not a verified quality-equivalent optimization. No training,
fresh generation, or GPU rental is needed to reuse the published library.
RunPod input validation rejects non-finite durations and oversized sample/step
counts before calling the GPU predictor; model execution uses inference mode.
