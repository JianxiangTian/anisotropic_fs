"""Verify release files using SHA-256, with LF normalization for text files."""
from _common import ROOT
import hashlib
import json
from pathlib import Path


def file_signature(path, normalization="none"):
    """Return the byte count and SHA-256 for the specified representation."""
    data = Path(path).read_bytes()
    if normalization == "lf":
        data.decode("utf-8")
        data = data.replace(b"\r\n", b"\n")
    elif normalization != "none":
        raise ValueError(f"Unknown normalization: {normalization}")
    return len(data), hashlib.sha256(data).hexdigest()


def verify_files(root, manifest):
    root = Path(root).resolve()
    errors = []
    for rel, expected in manifest.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(rel + ": missing/invalid path")
            continue
        try:
            size, digest = file_signature(path, expected.get("normalization", "none"))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{rel}: {exc}")
            continue
        if digest != expected["sha256"] or size != expected["bytes"]:
            errors.append(rel + ": content mismatch")
    return errors


def main():
    manifest = json.loads((ROOT / "provenance/release_manifest.json").read_text(encoding="utf-8"))
    errors = verify_files(ROOT, manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(manifest)} release files match the manifest.")


if __name__ == "__main__":
    main()
