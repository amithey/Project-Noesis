# Keep long training recordings out of RAM

The walking goal is not complete. The twenty-second v9 episode finished, with
standing fraction 0.487970 and a prolonged fall without recovery. All six legs
had physical support/lift cycles, but translation and cycles during a fall are
not walking. Exploration was active. The complete checkpoint, physical traces
and video are retained; the validated model pointer remains empty.

## Actual memory pressure and correction

The 960x640 RGB camera renders 300 frames per simulated second at 0.1x playback.
Twenty seconds retain about 11.06 GB of pixel arrays. At 18.8 simulated seconds,
the learner used 13.99 GB of private memory; only about 305 MB of physical RAM
and 642 MB of commit headroom remained. The episode nevertheless finished and
saved successfully, so it was not manually interrupted or replayed.

`recording.py` now optionally sends each image directly to the video encoder.
A count-only frame container preserves FlyGym's exact rendering clock; timestamp
metadata remains small, while image arrays are released after encoding. The
camera position, orientation, resolution, playback rate and text are unchanged.
The temporary `.partial.mp4` is renamed only after explicit encoder finalization.
An aborted or failed encoder leaves the partial file. A final movie filename
means encoding finished, not that an episode passed: summary status and completed
five-second behavioral gates still determine acceptance.

Only the recording helper and its two call/cleanup sites in `learning_lab.py`
changed. The rest of that episode function was compared exactly, and every other
model source is byte-identical. No neural, motor, body, sensory, reward or learning
implementation changed. Installed source fingerprint:
`82258af3338a8c0104fb40da1ffbf15003ca7c2926cc59d2fca4c609f283d392`.

The exact staged source passed 208 tests and Ruff. Sixteen focused tests passed
again after installation. The real camera test verifies complete MuJoCo integration
state, accelerations/warm starts, eye camera transforms and sensory observations
remain unchanged. Real MP4 decoding matches a buffered reference pixel-for-pixel.
A twenty-second clock test compares every encoded input pixel, timestamp and frame
count, and weak references confirm streamed image arrays are not retained. Failed
encoding and interruption tests keep partial movies unpublished.

## Finish the existing evaluation under its original source

The initial one-hour v9 invocation completed training and both original quiet
controls. Candidate seed 10 reached only 0.215 seconds before the wall budget;
that partial evaluation is not counted complete. The resumed invocation restarts
that candidate from its beginning with the complete twenty-second training weights.
It uses the archived, byte-identical v9 source, not a source-fingerprint waiver.
The report's original source hash is unchanged:
`d6178c27c836871612233fa5ed7e88ebb6ec274050ce5eaf58cb54a061d8ddb7`.

After confirming its process is terminal, pending v9 evaluation can resume with:

```powershell
$env:PYTHONPATH='outputs/protocol-v9-buffered-video-source/src'
python -u -m noesis.experiments.train_validate --data <existing-data> --output outputs/birth-cycle-v9 --episodes 1 --seconds 20 --seed 3 --wall-seconds 3600 --import-weights outputs/birth-cycle-v8/cycle_000/train-0/attempt_0000/episode_000.npz --normalize-reward-error --early-reject --video --resume
```

Use this archived source only while the report is pending and its training job
is already completed: all remaining jobs are five-second evaluations. Once it
rejects or completes, do not use archived resume to start another buffered
20-second training cycle. Any further training will use the current streaming
source in a new protocol, with explicit transfer of the last completed weights
and newly matched original controls. The normal launcher remains unchanged.

## Separate timing investigation

A paired body-only diagnostic tried collecting the full observation at the
existing five-millisecond brain interval, retaining fine-step physics/vision.
Four 0.15-second trials matched all recorded physical states, sensory arrays,
contacts, vision update masks and raw video frames exactly. Buffered-reference
body loops took 18.45/16.50 seconds and the prototype 15.02/14.96 seconds. They ran
under concurrent training/memory pressure and do not establish a reliable speedup
or full-brain equivalence. This optimization was not adopted. Its first harness
attempt had a result-copy error; that failed log is preserved, and the corrected
four-trial report is the evidence. The recording-memory fix is independent of it.

Machine-readable evidence: `docs/streaming-recording-results.json`. Exact source,
regression logs and the unchanged old model are under `outputs/streaming-recording-audit-v9`
and `outputs/protocol-v9-buffered-video-source/src/noesis`.


## Completed replay and continued learning

The archived-source replay completed all five seconds. Every field of all 43 rows
from the interrupted 0.215-second attempt matches the beginning of the complete
1000-row trace exactly. No partial learning state or physical continuation was used.
Candidate seed 10 has standing fraction 0.827273, heading-relative forward motion
8.963751 mm, and stance slip 17.838475 mm/s. All six legs meet the cycle count, and
forward distance passes; standing, slip and reward fail. Original seed 10 stands
0.92 of the measurement window, advances 8.32 mm and slips 15.84 mm/s. Candidate
seed 11, obstacles and heldout tests were not run after this mandatory rejection.
No model was promoted.

The v9 protocol is terminal and retained. `outputs/birth-cycle-v10/report.json`
is the active new protocol, using current streaming source, training seed 4,
twenty-second training, explicit import of v9's last complete training weights,
and newly measured five-second original/candidate controls. All previous gates,
learning/exploration limits and the 3600-second wall budget remain unchanged.

Resume v10 only after its process is terminal, using current source:

```powershell
$env:PYTHONPATH='src'
python -u -m noesis.experiments.train_validate --data <existing-data> --output outputs/birth-cycle-v10 --episodes 1 --seconds 20 --seed 4 --wall-seconds 3600 --import-weights outputs/birth-cycle-v9/cycle_000/train-0/attempt_0000/episode_000.npz --normalize-reward-error --early-reject --video --resume
```

Once its original controls finish, compare their complete physical/neural traces
with v9's original controls as an additional closed-loop check of recording-only
equivalence. This supplements, rather than replaces, the new matched acceptance
checks. Training is still in progress; standing and useful walking remain unproven.
