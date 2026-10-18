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
