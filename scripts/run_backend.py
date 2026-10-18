"""Start real local inference with the same credential file as the Docker frontend."""
import argparse
import os
from pathlib import Path
import sys

from dotenv import dotenv_values, load_dotenv
import uvicorn

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--action-config", default="configs/action_service_backend_smoke.yaml")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    config = (ROOT / args.action_config).resolve()
    if not config.is_file():
        parser.error(f"Action config not found: {config}")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    load_dotenv(ROOT / ".env")
    # Match Compose env_file semantics: the file wins over stale terminal keys.
    credentials = ROOT / ".env.compose"
    if credentials.is_file():
        key = dotenv_values(credentials).get("FORTEBANK_API_KEY")
        if not key:
            parser.error("Set FORTEBANK_API_KEY in .env.compose (shared with the frontend)")
        os.environ["FORTEBANK_API_KEY"] = key
        print("Using shared API key from .env.compose (value hidden)", flush=True)
    elif not os.environ.get("FORTEBANK_API_KEY"):
        parser.error("Set FORTEBANK_API_KEY in .env or .env.compose")
    os.environ.update(
        FORTEBANK_ANALYZER_FACTORY="overfitted_underqualified.backend.local_analyzer:create_analyzer",
        FORTEBANK_ACTION_CONFIG=str(config),
        FORTEBANK_ENABLE_MOCK_RESULTS="false",
    )
    uvicorn.run("overfitted_underqualified.backend.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
