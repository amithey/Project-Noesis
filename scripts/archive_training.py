"""Package training evidence from all registered worktrees for a GitHub release.

Run: python scripts/archive_training.py --output outputs/github-sync/artifacts
Archives retain paths relative to each checkout. Raw data and derived connectome
caches are deliberately excluded. Upload the ZIPs and manifest.json to a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXTENSIONS = {".json", ".npz", ".mp4", ".png"}
EXCLUDED = {"data", "raw", "cache", "brain-cache", "gemini-package", "github-sync"}


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], encoding="utf-8"
    ).strip()


def eligible(path, root):
    relative = path.relative_to(root)
    parts = {part.lower() for part in relative.parts[:-1]}
    return (
        path.suffix.lower() in EXTENSIONS
        and not parts.intersection(EXCLUDED)
        and not any("cache" in part for part in parts)
        and path.stem.lower() not in {"neurons", "connections", "connectome"}
        and not path.is_symlink()
    )


def package(root, output):
    branch = git(root, "branch", "--show-current") or "detached"
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", branch)
    archive = output / f"{name}.zip"
    roots = [root / "outputs", root / "gpu-pilot" / "outputs"]
    files = sorted({p for folder in roots if folder.exists()
                    for p in folder.rglob("*") if p.is_file() and eligible(p, root)})
    if not files:
        return None
    entries = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as bundle:
        for path in files:
            before = path.stat()
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as source, bundle.open(relative, "w", force_zip64=True) as dest:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    dest.write(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"File changed during backup: {path}; retry after writer stops")
            entries.append({"path": relative, "bytes": after.st_size,
                            "sha256": digest.hexdigest()})
        bundle.writestr("ARTIFACT-MANIFEST.json", json.dumps(entries, indent=2))
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"branch": branch, "source_commit": git(root, "rev-parse", "HEAD"),
            "archive": archive.name, "bytes": archive.stat().st_size,
            "sha256": digest, "files": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    archives = []
    for line in git(root, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            tree = Path(line[9:])
            result = package(tree, args.output)
            if result:
                archives.append(result)
                print(f"Packaged {result['archive']}: {len(result['files'])} files", flush=True)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Training NPZ, JSON reports, MP4 videos and PNG previews; no raw data or connectome caches",
                "archives": archives}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(archives)} archives and manifest.json", flush=True)


if __name__ == "__main__":
    main()
