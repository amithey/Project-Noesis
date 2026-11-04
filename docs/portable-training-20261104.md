# Run Noesis on a second computer

The [portable training release](https://github.com/amithey/Project-Noesis/releases/tag/portable-training-2026-11-04) contains a small bootstrap archive and the complete eligible training and diagnostic outputs from the primary FlyGym 2 pilot. Download the bootstrap alone to start. Download the full evidence archive for observations, reports, logs, videos and other checkpoints. Both preserve original paths and per-file SHA-256 records. Extract one archive into a new directory.

- Pilot runtime: `75cdc34aeeacb7684c73b6e91bd39fddf399cc9f`.
- Core runtime: `42c20769376bda5694b4c1c66ca75e52cbd9f1f4`.
- Starting bank: completed experimental parent `9fd7d411…`. The bootstrap also includes experimental C2 `d7013c88…` and historical standing ancestor `fb80cfcc…`.
- A partial bank is evidence of an interrupted episode. The project has no validated-best bank and starts a fresh episode from a completed checkpoint.

## Download while the release is a draft

Sign in to GitHub with an account that can edit this repository and open the
[draft release](https://github.com/amithey/Project-Noesis/releases/tag/untagged-c48beb90cc6d01c18b87).
Download `noesis-bootstrap.zip` and `manifest.json` into a local `downloads/` folder.
Download `noesis-pilot-evidence.zip` as well if you want the complete pilot evidence.
The tag-based download command below applies after publication. Skip that line
when you already downloaded the files from the draft.

## Start the agent

Download the goal package from the same draft. Open the restored project in Codex and paste the complete contents of [noesis-agent-goal.md](noesis-agent-goal.md) into the new project chat as a goal. Read the current coordination files before work.

## Install on Windows

Use an empty working directory, Git, GitHub CLI and Python 3.12. Replace the data path below with your actual path.

```powershell
git clone https://github.com/amithey/Project-Noesis.git noesis-pilot
git -C noesis-pilot checkout --detach 75cdc34aeeacb7684c73b6e91bd39fddf399cc9f
git clone https://github.com/amithey/Project-Noesis.git noesis-core
git -C noesis-core checkout --detach 42c20769376bda5694b4c1c66ca75e52cbd9f1f4

py -3.12 -m venv .venv
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
& $python -m pip install -e '.\noesis-core' -e '.\noesis-pilot\gpu-pilot[dev]'
& $python -m pip check

gh release download portable-training-2026-11-04 -R amithey/Project-Noesis --pattern noesis-bootstrap.zip --pattern manifest.json --dir downloads
$manifest = Get-Content downloads\manifest.json -Raw | ConvertFrom-Json
$archive = $manifest.archives | Where-Object name -eq 'noesis-bootstrap.zip'
if ((Get-FileHash downloads\noesis-bootstrap.zip -Algorithm SHA256).Hash -ne $archive.sha256) { throw 'Archive SHA-256 mismatch' }
Expand-Archive downloads\noesis-bootstrap.zip -DestinationPath restored
foreach ($item in $archive.files) {
    $file = Join-Path restored $item.path
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $item.sha256) { throw "File SHA-256 mismatch: $file" }
}
$parent = (Resolve-Path 'restored\gpu-pilot\outputs\e4-exploration-20261029\train-control-1000\weights-last.npz').Path
$core = (Resolve-Path 'noesis-core\src').Path
$data = 'D:\NoesisData'
```

Populate the licensed connectome dataset separately using the pinned core's data instructions. The expected raw FlyWire inputs include `data/raw/neurons.csv` and `data/raw/connections_princeton.csv`. Use the matching data version and layout. The release excludes licensed raw datasets, derived caches and local virtual environments.

## Check the machine before training

Register the host, output directory, seed and time budget. Use a single heavy simulation per host and the project's canonical lease on a shared host. First complete a short frozen CPU probe with a new output path. Inspect `report.json` for the planned chunk count, completion status and unchanged weights. Allow up to 600 seconds for this check and stop on a technical failure.

```powershell
& $python -m noesis_gpu.run --backend cpu --worlds 1 --seconds 0.1 --wall-seconds 600 --noesis-source $core --data $data --checkpoint $parent --seed 10006 --output runs\transfer-smoke
```

After a successful smoke check, register a matched experiment before training. The following is the existing five-second CPU training entry point with a separate output directory. Set the experiment's objective and other factors explicitly when matching a registered protocol. Allow at most 600 seconds for this example. Stop on timeout, failure, or an incomplete episode and retain partial evidence separately.

```powershell
& $python -m noesis_gpu.train --backend cpu --seconds 5 --wall-seconds 600 --noesis-source $core --data $data --checkpoint $parent --seed 1000 --output runs\transfer-train-1000
```

Preserve the same original reference bounds and topology. Keep the final validation seeds 10, 11, 200, 201, 202 out of development. Use the pilot README for NVIDIA body checks and CPU/GPU parity before changing the backend. Report the actual source hashes and environment with new runs.

## Historical evidence

- [October 27 worktree backup and archive manifest](https://github.com/amithey/Project-Noesis/releases/tag/training-backup-2026-10-27)
- [Historical gait candidates and videos](https://github.com/amithey/Project-Noesis/releases/tag/gait-validation-stage2-stage3-2026-10-28)
- [All experiment Releases, including Claude's chain diagnostic](https://github.com/amithey/Project-Noesis/releases)

Keep historical archive extracts separate. Report paths from the original computer are provenance and should be remapped for new commands.
