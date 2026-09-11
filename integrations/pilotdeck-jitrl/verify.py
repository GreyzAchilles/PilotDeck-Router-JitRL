from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PilotDeck JitRL integration patch.")
    parser.add_argument("pilotdeck", type=Path, help="Path to a PilotDeck checkout")
    parser.add_argument(
        "--apply-check",
        action="store_true",
        help="Run git apply --check. The checkout must not already contain the integration.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    patch = root / "pilotdeck-jitrl.patch"
    overlay = root / "overlay"
    pilotdeck = args.pilotdeck.resolve()

    if not patch.is_file() or patch.stat().st_size == 0:
        raise SystemExit("missing or empty pilotdeck-jitrl.patch")
    if not overlay.is_dir():
        raise SystemExit("missing overlay directory")
    if not (pilotdeck / ".git").exists():
        raise SystemExit(f"not a Git checkout: {pilotdeck}")

    overlay_files = sorted(path for path in overlay.rglob("*") if path.is_file())
    mismatches: list[str] = []
    for source in overlay_files:
        relative = source.relative_to(overlay)
        target = pilotdeck / relative
        if not target.is_file():
            mismatches.append(f"missing: {relative.as_posix()}")
            continue
        if sha256(source) != sha256(target):
            mismatches.append(f"different: {relative.as_posix()}")

    print(f"patch: {patch} ({patch.stat().st_size} bytes, sha256={sha256(patch)})")
    print(f"overlay files: {len(overlay_files)}")
    if mismatches:
        print("overlay verification failed:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        return 1
    print("overlay verification: PASS")

    if args.apply_check:
        subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=pilotdeck,
            check=True,
        )
        print("git apply --check: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
