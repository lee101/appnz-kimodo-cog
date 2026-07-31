from __future__ import annotations

import json

from runpod_batch import TAG, gql


def main() -> None:
    query = """query { myself { pods { id name desiredStatus runtime { uptimeInSeconds } } } }"""
    pods = gql(query)["myself"]["pods"] or []
    tagged = [pod for pod in pods if (pod.get("name") or "").startswith(TAG)]
    print(json.dumps({"count": len(tagged), "pods": tagged}, indent=2))


if __name__ == "__main__":
    main()

