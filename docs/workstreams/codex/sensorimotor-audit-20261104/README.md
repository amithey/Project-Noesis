# LSA + NIR: native load range and direct neural input observability (2026-11-04)

These are **offline diagnostics on existing development recordings**, not new training or proof of walking. Runtime, connectome, weight signs/bounds, checkpoints, gates and final held-outs were unchanged. Codex owns these new review-only files; Claude retains sensory/learning implementation ownership.

## LSA: almost-constant load saturation is not supported in the native body

Registration: PR24comment6635445880. All12S1 archived frozen trajectories (parent/control/child; seeds10006/10007/10010/10011),1000rows each, first10rows excluded. The registered saturation definition was tanh(load/load0)>=.99; thresholds were not fitted to outcomes.

At the actual load0=2 setting:
- Foot-contact leg frames exceed the saturation threshold in **10.3%–24.0%**, not almost all frames.
- Median whole-leg contact-load norm on foot-contact frames is **2.70–3.40 native force units** across cases.
- The previously suggested load0=45 produces zero saturated contact frames in a counterfactual transfer calculation. That alone does not make it a better sensor: reducing gain can discard useful contrast. No neural/body run used45.

The prior broad claim that adhesion makes load afferents essentially constant at150Hz does not transfer to these recorded native FlyGym2 trajectories. This does not rule out saturation in particular legs/times, any neural-network operating-point problem, or a useful calibration at a different scale.

Source: backend.observe computes whole-leg load from the norm of recorded whole-leg ground-contact sensor force components, separately from tarsus5 foot_load. SensoryMap uses reconstructed rate=2+148*tanh(load/2) for campaniform/bristle groups. Native force units are not biologically calibrated here. Observations are end-of-step states that feed the next control step; reconstructed transfer values are not a log of actually sampled sensory spikes, and the final recorded state has no subsequent step in that episode.

Audit file SHA256 values, per-leg saturation, force/rate quantiles and normalized derivatives are in load-range-result.json. No raw neuron/connectome tables were published.

## NIR: a weak direct input-to-firing association is visible before anatomical pooling

Registration: PR26comment6635476098. Exploratory reanalysis of Claude's existing chain archive, already inspected at the pooled-result level. This is **not an independent confirmatory study**.

Input archive SHA256:
c4d762c4a198b4a53679dc93b9de77c44a586a5ca1b96a323e72afc50fc45084.

All four driver reports show complete5s/three-arm runs, correct parent9fd7d411, equal initial weight contexts. The reanalysis requires1000finite rows, exact off/zero invariance, equal target ordering/rest arrays and zero off/zero currents. These checks passed before interpretation.

Primary: Pearson r between each target's injected current and that SAME target's within-bin spike count, at the fixed lag0; average r over the **common varying-target set across all three arms**. No lag search or anatomy-based selection. Constant-response channels are excluded explicitly. The reference shifts every neuron column by the same randomly sampled200..800bins,1000draws,seed0, preserving cross-neuron structure. Reference ranks remain descriptive because circular-shift stationarity/calibration has not been established.

| Dev seed | Common valid targets /453 | Active mean r | Off=zero mean r | Active positive-direction reference rank | Off=zero rank |
| --- | ---: | ---: | ---: | ---: | ---: |
|10006|313|0.03035|-0.00448|1/1001|0.769|
|10007|331|0.03247|0.00991|1/1001|1/1001|
|10010|334|0.02761|-0.01418|1/1001|1.000|
|10011|323|0.03342|0.00259|1/1001|0.413|

Active mean r exceeds its matched off/zero value on all4seeds. This shows a weak input-associated signature in the recorded targeted firing under this analysis, despite the earlier signed joint-pooled statistic not meeting its rule. It does not establish downstream transmission, useful movement, learning, retained standing, or why previous training failed. The four seeds are the replication units; hundreds of neurons are not independent replicate flies.

**Important limitations:**119–140targets per seed are excluded by the common variance mask; the conclusion does not cover silent/constant targets. Off and zero are identical, not independent controls. One negative-control seed also gets the minimum descriptive rank: never treat the active rank alone as calibrated significance or proof of causality. The alternative analysis was chosen after previous results were known. Original registered chain verdicts remain unchanged. No claim of a broken chain or absent learning follows from either analysis.

## Checks and next decision

The NIR script verifies its FFT orientation against direct time-domain correlation and a perfectly coupled synthetic control. Those are implementation checks, not physical evidence. Both scripts completed below their separately registered60s offline budgets, with full input hashes in their JSON output.

Decision: do not implement load0=45 as an assumed fix and do not increase exploration just to force a larger signal. Before choosing a new learning intervention, Claude's next proposed downstream/credit analysis should retain neuron/joint heterogeneity, test negative controls and the actual credit/reward timing, and explicitly distinguish direct neural observability from useful motor control. PR27's fail-closed evidence/timeout repairs remain separate owner tasks. Any physical calibration/learning experiment still needs its own registered controls and retention criteria; none is allocated by these diagnostics.

Full goal remains unmet: stable fly-like walking and environmental response that improve and persist through training. B2 baseline qualification permits matched experiments on this host; it does not promote any checkpoint.
