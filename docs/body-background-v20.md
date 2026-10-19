# Frozen body background-drive diagnostic

V18's local motor-adaptation hypotheses were rejected and remain uninstalled.
V19 then recorded full-network activity and proved exact production/prototype
default equivalence for all six specified model/case pairs. Background-only
motor activity persists without a body or sensory feedback. This motivates
probing the existing background input gain directly in the body; it does not
prove that this input caused the fall or that lower gain will improve gait.

Eight complete five-second quiet trials compare original/final V17 weights,
seeds 66/67 and background event weights 60/40 at the same 3 Hz rate. Each birth
starts fresh; trained cases explicitly import only compatible weights from the
unchanged final V17 bank (8755d9888de39337cf4286526a92ca6568d38bca8deb9e965f16cb0f5a1713fc).
Learning, motor exploration and training-fall termination are all off. Vision,
touch, proprioception and balance remain on. The muscles body, geometric
adhesion, neural wings, U=0, da=1, legacy encoding and v4 reward are unchanged.
There is no external gait clock, leg sequence or joint controller addition.

The entire production source remains exact V17:
839a0a8bd85a8ebca55048319dc95642413491a9a14c5d168fe24cc4d6e8d107.
Only existing run arguments differ. Model-context bg_hz/bg_weight values,
zero learner traces, unchanged original/trained weights, source/plan hashes
and the complete five-second window are checked. Seventeen argument/accounting
contract tests pass. Actual archived frozen checkpoints also pass the new
verifier at matching gain and are rejected at a mismatched gain. This caught
and corrected an initial assumption about context field names before launch.

The first original-weight seed66/background60 trial has started with learning
off. Observe only process stdout while live; read/copy the ledger after verified
termination. Each worker has an absolute 3,600-second deadline. Incomplete cases
remain outside completed jobs and replay from their complete start weights.

Report the parameter contrast at fixed weights separately from the learned
weight contrast at the same background gain. Every full trial retains the
standing, heading progress, leg-cycle, slip and reward metrics plus video,
including falls. Immobility or a lower all-window slip after falling is not
walking. Diagnostic seeds66/67 cannot replace validation10/11, heldout200..202
or obstacle/ablation controls. There is no automatic promotion or startup change
regardless of diagnostic results. Learned walking remains unproven.

## First hour completed; remaining cases resumed (2026-10-19)

The first worker exited normally at its wall budget with five complete cases and
one incomplete one-second case. Its terminal report SHA256 is
c51e3bb60eff35d214d54c98ba406a0ef09a690a0b14ecbb24ec5371440566b0.
No matching Python worker remained in the elevated OS process inventory before
copying that report. All five complete cases were independently checked against
source/plan/model-context/weight hashes and copied with their videos into the
user workspace. The shortened sixth case was preserved separately and never
entered completed jobs. Exact resume started a fresh sixth-case attempt from the
unchanged V17 parent bank, within another 3600-second budget (session92000).
Observed neural/body stdout through1.2s confirms the resumed worker is live.
The first0.8s samples match the interrupted trial's printed body/neural values.
A full artifact-prefix comparison remains pending completion.

| Weights | Seed | Background | Standing fraction | All-window stance slip mm/s |
| --- | ---: | ---: | ---: | ---: |
| Original | 66 | 60 | .789899 | 12.620769 |
| Original | 66 | 40 | .957576 | 15.429347 |
| Original | 67 | 60 | .322222 | 10.400567 |
| Original | 67 | 40 | .613131 | 12.609414 |
| V17 trained | 66 | 60 | .060606 | 5.970574 |

The lower background gain improved standing in both original-weight diagnostic
seeds, but seed67 still fails90%. These are parameter contrasts, not evidence
that weight learning improved gait. Neither contrast reduces slip.

The independent `audit_copied_csv.py` verifies copied CSV hashes, all1000 body
samples, the990 awake samples and exact5ms cadence. It independently reconstructs
standing, mean slip, reward and heading displacement, agreeing with all five
summaries. Its additional conditional measurements do not change acceptance:
for trained seed66/background60, standing-window slip is13.622492mm/s against
original13.998776 (about2.7% lower), whereas all-window slip misleadingly appears
52.7% lower. Trained heading progress during standing is -0.395072mm, versus
+6.321267mm original; its all-window+8.784498mm is mostly outside standing.
This is failure evidence, not learned walking. Original background40 seed66 has
standing-window slip15.415065mm/s. Seed67 is14.767169at60 vs14.968752at40.

`preserve_completed_case.py` consumes stdout-confirmed immutable case receipts
without opening the live report. The CSV auditor similarly reads only copied,
completed artifacts. Ruff passes for both helpers. Production source and the
running driver/plan remain unchanged. No model promoted, startup unchanged;
three complete frozen diagnostic cases and all later required gates remain.
## Sixth case complete; gain reduction does not rescue the trained bank

The resumed seed66/background40 trained case completed five seconds. Standing
is79.4949% versus95.7576% in its matched original control; all-window slip is
16.007278 versus15.429347mm/s, so both standing and relative-slip gates fail.
Standing-conditioned slip is14.689525 versus15.415065mm/s (about4.7% lower,
not20%). Forward during standing is6.068745mm; all-window3.376297mm. These are
useful movements but do not establish accepted walking. The case and all its
immutable artifacts were independently checked/copied; the live ledger was
not opened. Six complete CSVs now pass the independent physical-summary audit.

The replay reproduces all200 interrupted CSV rows exactly across every column,
and the first8594 motor events have identical times/indices (partial950neuralms
versus4950ms completed). Both frozen checkpoints have identical SHA256
cd49364cc80d8923a74d7783d696fd1a90a132d53afa344610d70663b8a11b19 and zero
learning traces. The one-second attempt remains incomplete and separate.

Three frames from original seed66/background40 were directly viewed at displayed
physical times1.52,3.02,4.52 seconds; sampled bodies are upright and their leg
configurations differ. Sparse frames do not certify continuous support or gait,
and this is explicitly an original-weight control. The image/review are saved.
The live worker has advanced to trained seed67/background60, with two cases
remaining. A separate earlier-training-bank test has been prepared, not launched;
see best-training-v21.md. All source/driver/plan hashes for V20 remain unchanged.