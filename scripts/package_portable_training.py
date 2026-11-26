"""Package and verify portable Noesis training artifacts for a GitHub Release."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

EXTENSIONS = {".npz", ".json", ".mp4", ".png", ".log", ".txt"}
EXCLUDED = {"data", "raw", "runtime", "arrow-runtime", "venv", ".venv", ".git"}
BANKS = {
    "e4-exploration-20261029/train-control-1000": "9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c",
    "credit-replication-c2-20261030/train-1002": "d7013c88af9aa5a20e150f2aad6e2d4395aa2c58603ada775016e470393d1dc1",
    "standing-001-train-11000": "fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def eligible(path, source):
    parts = [part.lower() for part in path.relative_to(source).parts]
    return (
        path.is_file() and not path.is_symlink() and path.is_relative_to(source)
        and path.resolve().is_relative_to(source)
        and path.suffix.lower() in EXTENSIONS
        and not any(part in EXCLUDED or "cache" in part or part.startswith(".") for part in parts)
        and not any(word in path.stem.lower() for word in ("credential", "secret", "token", "connectome"))
    )


def archive(source, files, destination):
    records = []
    with ZipFile(destination, "x", ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(files):
            before = path.stat()
            name = "gpu-pilot/outputs/" + path.relative_to(source).as_posix()
            sha256 = digest(path)
            bundle.write(path, name)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"File changed during backup: {path}")
            records.append({"path": name, "bytes": after.st_size, "sha256": sha256})
        bundle.writestr("ARTIFACT-MANIFEST.json", json.dumps(records, indent=2))
    with ZipFile(destination) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError(f"CRC verification failed: {destination}")
        for record in records:
            with bundle.open(record["path"]) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch: {record['path']}")
    return {
        "name": destination.name, "bytes": destination.stat().st_size,
        "sha256": digest(destination), "file_count": len(records), "files": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = (args.project / "gpu-pilot/outputs").resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(source):
        parser.error("Output must be outside the source tree")
    files = [path for path in source.rglob("*") if eligible(path, source)]
    bootstrap = []
    for folder, expected in BANKS.items():
        directory = source / folder
        if digest(directory / "weights-last.npz") != expected:
            raise RuntimeError(f"Unexpected bootstrap checkpoint: {directory}")
        for name in ("weights-last.npz", "weights-last.model.json", "report.json"):
            path = directory / name
            if path not in files:
                raise RuntimeError(f"Missing bootstrap file: {path}")
            bootstrap.append(path)
    output.mkdir(parents=True, exist_ok=False)
    bundles = [
        archive(source, bootstrap, output / "noesis-bootstrap.zip"),
        archive(source, files, output / "noesis-pilot-evidence.zip"),
    ]
    manifest = {
        "scope": "Primary checkout gpu-pilot/outputs training and diagnostic evidence. Historical worktrees use existing Releases.",
        "pilot_commit": "75cdc34aeeacb7684c73b6e91bd39fddf399cc9f",
        "core_commit": "42c20769376bda5694b4c1c66ca75e52cbd9f1f4",
        "bootstrap_banks": BANKS,
        "archives": bundles,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for bundle in bundles:
        print(bundle["name"], bundle["file_count"], bundle["bytes"], bundle["sha256"])


if __name__ == "__main__":
    main()
