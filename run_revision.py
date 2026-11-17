"""Download and verify the pinned data, then run the exact revised experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
from data import download

ROOT = Path(__file__).resolve().parent
REVISION = ROOT / "revision"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=["controls", "tuning", "spatial", "gateways", "counts", "quality"],
                        default=["controls", "tuning", "spatial", "gateways", "counts", "quality"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results/revision_v1")
    parser.add_argument("--check-only", action="store_true", help="Verify source and pinned data without training")
    args = parser.parse_args()
    download(REVISION / "data")
    manifest = json.loads((REVISION / "source_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((REVISION / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source/data does not match the published experiment snapshot: " + name)
    print("Verified exact revision source and data hashes.", flush=True)
    if args.check_only:
        return
    subprocess.run([sys.executable, str(REVISION / "ieee_access_resubmit/experiments.py"),
                    "--jobs", str(args.jobs), "--output", str(args.output.resolve()),
                    "--stages", *args.stages], check=True)


if __name__ == "__main__":
    main()
