"""Recompute and package the terminal LR4 evidence from registered inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from run_load_range_lr4 import OBS_HASHES, REPORT_HASHES, SEEDS, digest, read, summarize


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    parent = root / "load-range-lr4-20261105"
    result = read(parent / "result.json")
    expected = [(seed, scale) for seed in SEEDS for scale in (2, 4)]
    jobs = result["jobs"]
    if result["status"] != "completed" or [(j["seed"], j["scale"]) for j in jobs] != expected:
        raise ValueError("Require all registered completed arms in order")
    if not 0 < result["wall_seconds"] <= 1450:
        raise ValueError("Aggregate budget receipt mismatch")

    archive = {"result.json": (parent / "result.json").read_bytes()}
    windows = {}
    for job in jobs:
        seed, scale = job["seed"], job["scale"]
        index = SEEDS.index(seed)
        name = f"scale-{scale}-{seed}"
        run = parent / name
        reference = root / f"interface-f10-20261105/frozen-{seed}"
        if (digest(reference / "report.json") != REPORT_HASHES[index]
                or digest(reference / "observations.npz") != OBS_HASHES[index]):
            raise ValueError("Archived F10 evidence hash mismatch")
        recomputed = summarize(run, reference, seed, scale)
        if any(job[k] != v for k, v in recomputed.items()):
            raise ValueError(f"Summary mismatch: {name}")
        windows[seed, scale] = recomputed["windows"]
        raw = (run / "observations.npz").read_bytes()
        if sha(raw) != job["observation_sha256"]:
            raise ValueError(f"Observation hash mismatch: {name}")
        archive[f"{name}/observations.npz"] = raw
        report = read(run / "report.json")
        for field in ("noesis_source", "checkpoint"):
            report.pop(field, None)
        report["local_paths_removed"] = True
        archive[f"{name}/report.json"] = encode(report)
        telemetry = read(parent / f"{name}.telemetry.json")
        if (telemetry["exit_code"] != 0 or telemetry["final_exit_code"] != 0
                or telemetry["process_wall_seconds"] != job["process_wall_seconds"]
                or not 0 < telemetry["process_wall_seconds"] <= 330
                or not telemetry["samples"]
                or min(row["host_available_gib"] for row in telemetry["samples"]) < 1):
            raise ValueError(f"Telemetry mismatch: {name}")
        preflight = read(parent / f"{name}-preflight.json")
        samples = preflight["samples"]
        if (not preflight["decision"]["passed"] or len(samples) != 10
                or min(row["available_gib"] for row in samples) < 2
                or sum(row["host_cpu_percent"] for row in samples) / 10 > 60):
            raise ValueError(f"Admission mismatch: {name}")
        archive[f"{name}.telemetry.json"] = encode(telemetry)
        archive[f"{name}-preflight.json"] = encode(preflight)

    comparisons = []
    for seed in SEEDS:
        early, late = [
            dict(standing_delta=b["standing"] - a["standing"],
                 slip_delta_mm_s=b["slip_mm_s"] - a["slip_mm_s"])
            for a, b in zip(windows[seed, 2], windows[seed, 4], strict=True)
        ]
        passed = (early["standing_delta"] >= 0 and late["standing_delta"] >= .05
                  and early["slip_delta_mm_s"] <= 0 and late["slip_delta_mm_s"] <= 0)
        comparisons.append(dict(seed=seed, early=early, late=late, passed=passed))
    if (comparisons != result["comparisons"]
            or result["exploratory_pass"] != all(row["passed"] for row in comparisons)):
        raise ValueError("Preregistered verdict mismatch")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-outputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive = verify(args.pilot_outputs)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {name: {"bytes": len(data), "sha256": sha(data)} for name, data in archive.items()}
    path = args.output / "lr4-evidence.zip"
    with zipfile.ZipFile(path, "x", zipfile.ZIP_STORED) as bundle:
        for name, data in archive.items():
            bundle.writestr(name, data)
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None or any(sha(bundle.read(n)) != m["sha256"] for n, m in manifest.items()):
            raise ValueError("Archive verification failed")
    index = dict(files=manifest, zip_bytes=path.stat().st_size, zip_sha256=digest(path))
    (args.output / "lr4-manifest.json").write_bytes(encode(index))
    print(json.dumps({k: v for k, v in index.items() if k != "files"}))


if __name__ == "__main__":
    main()
