# A PN read-out trained from reward only (no side labels), tested with the real network — 2026-10-26

Branch `feature/network-calibration`; code `noesis.reward_readout`, `noesis.experiments.reward_readout_run`. Same leg-free point agent and wide-antenna
world (3 mm), calibrated network (`g_I` = 12, graded APL 1024), PN features = log(1 + 250 ms average rate) of 699 projection neurons.

## Method
- **Read-out:** score = clip(w · z + exploration noise, -1, 1), z = 20 unsupervised principal components of the recorded PN features
  (no labels, no reward used to choose them). The agent turns left when the score is positive. Nothing in training tells the read-out
  which side was stronger.
- **Learning:** REINFORCE with a batch-mean baseline, SGD with momentum; episode return = fraction of the starting distance closed
  (+1 for reaching the food, + up to 0.5 for reaching it early), **multiplied by the hunger factor (1.7 hungry, 1.0 fed)**.
- **Why a surrogate:** reinforcement learning needs thousands of episodes and one episode with the real network takes ~3 min. Training therefore
  ran on a library of PN feature vectors RECORDED from the real network under constant inputs (168 blocks: 6 concentration levels x 11
  left/right ratios, including near-1 ratios recorded for this experiment); each control step the antenna concentrations were converted to an
  effective level and ratio and a recorded vector from the nearest block was drawn. The surrogate has no network dynamics; hungry gain also
  multiplies the effective level (as in the real network, where receptor rate scales with the gain).
- **Hyper-parameters** (PCA components 20, learning rate 0.1, noise 0.5, 300 iterations x 16 episodes, L2 0.01) were chosen from a small sweep
  on surrogate episodes only; no closed-loop result was used to tune anything.
- **Closed-loop test:** learned weights frozen, real network in the loop, 12 start poses (seeds 0-11, as in the earlier label-decoder experiment),
  12 s episodes. Pre-registered main comparison: `reward_hungry` against each control (one-sided Fisher on reaching the food, Holm-corrected; exact sign
  test on final distance). Controls: shuffled reward (return assigned from another episode), a random read-out of the same spread, reversed sign,
  no gradient (both antennae sample the body centre), no steering; the earlier side-label decoder is a reference.

## Surrogate training (300 fresh surrogate episodes each; not evidence about the real network)
| Model | success | median final distance | learning curve (success every 50 iterations) |
|---|---|---|---|
| hungry | 76% | 2.9 mm | 0.19, 0.14, 0.61, 0.80, 0.78, 0.76 |
| fed | 83% | 2.9 mm | 0.32, 0.50, 0.86, 0.78, 0.74, 0.77 |
| shuffled reward | 9% | 29 mm | 0.36, 0.21, 0.43, 0.38, 0.03, 0.10 |
| random read-out | 1% | 33 mm | — |
Reward-only learning works in the surrogate and is absent with shuffled reward; hunger did not speed it up (fed was as fast or faster).

## Closed loop with the real network (reached the food, of 12 poses)
| Condition | reached | Fisher vs reward_hungry (Holm) | reward_hungry ended closer to the food than this condition in n of 12 poses |
|---|---|---|---|
| **reward_hungry** (main) | **2** | — | — |
| reward_fed (own weights, fed gain) | 5 | not an a priori comparison (hunger goes the wrong way) | 3 |
| shuffled_reward | 0 | p = 0.24 (1.00) | 7 |
| random_readout | 1 | p = 0.50 (1.00) | 10 |
| reversed_sign | 0 | p = 0.24 (1.00) | 7 |
| no_gradient | 0 | p = 0.24 (1.00) | 9 |
| diffusion_only | 0 | p = 0.24 (1.00) | 12 (sign test p < 0.001: it simply leaves the arena) |
| label_decoder (reference, trained with side labels) | 9 | — | 0 |

Final distance (mm) per pose, hungry model: 33 12 29 44 18 40 12 **3** 43 37 32 **3**; fed: 8 57 27 **3 3 3 3** 26 13 **3** 12 19; label decoder: reached 9 of 12 (replicates the earlier 9/12).
- **The pre-registered main comparison is negative:** the hungry reward-trained read-out reached the food in 2 of 12 poses, no better than any control after
  correction (0-1 of 12).
- **Exploratory, after seeing the data:** the fed model reached the food in 5 of 12 against 1 of 60 pooled control episodes (one-sided Fisher p = 0.0003).
  This is post hoc, the controls were built for the hungry weights (not the fed ones), and the pooled controls mix different mechanisms,
  so it is a hint that a reward-trained read-out can steer in the real network, not a result. The hungry model, trained the same way, did not
  (2 of 12; p = 0.07 against the pooled controls).
- **The label-trained read-out remains clearly better** (9 of 12), as in the earlier experiment.
- **Hunger did not help**; the direction is opposite to the hypothesis (fed 5 of 12, hungry 2 of 12; a pose-by-pose sign test is not significant).

## Reading and limits
- Learning from reward alone, without side labels, worked on recorded PN data but transferred only weakly (fed) or not at all (hungry) to the real
  network. Likely contributors (not tested): the surrogate ignores the network's temporal dynamics and the lag of the 250 ms average; the score
  distribution of the real loop is noisier (score sd about 0.8 in a smoke test); a narrower mismatch at the high receptor gain of the hungry state
  (effective level up to 2.9).
- 12 poses; one training seed per model; hyper-parameters tuned on the surrogate; control weights matched only to the hungry model.
- Even a success would be a trained external read-out of projection neurons (one synapse from the receptors), not learning inside the network.

## Follow-up: fine-tuning with the real network in the loop (48 episodes per arm)
Question: is the gap between the surrogate (76% success) and the real network (2 of 12) caused by the surrogate? Test: start from the hungry
surrogate weights and continue REINFORCE with the REAL network in the loop (`reward_readout_run finetune`): 6 iterations x 8 episodes of 12 s,
random start poses (never the 12 evaluation poses), exploration noise 0.5 on the score, learning rate 0.2, momentum 0.5, return x 1.7
(hunger). Control arm: identical poses, identical start weights, but each episode's return is replaced by another episode's return of the same
iteration (reward contingency broken). The learned weights were then frozen and evaluated on the same 12 poses as everything else.

| Quantity | reward arm | shuffled-return arm |
|---|---|---|
| reached the food during training, per iteration of 8 | 2, 1, 3, 1, 2, 0 | 2, 3, 0, 4, 3, 3 |
| mean return per iteration | 0.79, 0.75, 1.07, 0.41, 0.93, 0.30 | — |
| change of the weight vector from the surrogate start (start norm 1.70) | 3.21 | 0.95 |
| frozen evaluation, reached the food of 12 poses | **3** | **1** |
Frozen evaluation per pose (final distance mm; * = reached): reward arm 4.5, 36, 40, 48, 2.7*, 2.9*, 2.9*, 28, 26, 14, 28, 40; shuffled arm 31, 40, 39, 12, 23, 52, 33, 16, 2.9*, 47, 40, 19.

Comparison on the 12 evaluation poses (reached the food; one-sided Fisher against the fine-tuned reward arm, Holm-corrected over the seven
controls):
| Condition | reached | p (Holm) |
|---|---|---|
| fine-tuned reward arm | 3 | — |
| surrogate-only hungry read-out (before fine-tuning) | 2 | 0.50 (1.00) |
| fine-tuned shuffled-return arm | 1 | 0.30 (1.00) |
| random read-out | 1 | 0.30 (1.00) |
| shuffled reward (surrogate), reversed sign, no gradient | 0, 0, 0 | 0.11 (0.87) each |
| diffusion only | 0 | 0.11 (0.87) |
| surrogate-only fed read-out | 5 | 0.90 (1.00) |
| label-trained read-out (reference) | 9 | — |

- **Pre-registered criterion not met.** The fine-tuned reward arm (3 of 12) did not reach the food significantly more often than its shuffled-return
  control or any other control, and its improvement over the surrogate-only model (2 of 12) is one pose.
- During real-network fine-tuning there was no upward trend in success per iteration (2, 1, 3, 1, 2, 0 of 8) and the reward arm's weights drifted more
  than three times as far as the control's (3.2 versus 0.95) without a behavioural gain; the last iteration was the worst.
- **What this does and does not show.** It does not show that the simulator gap is the cause: 48 real episodes (about 2.6 simulated
  hours of agent time) with a 20-weight read-out and a high-variance estimator is too little experience to learn from noisy returns. It also does
  not show that reward learning cannot work in the real loop; the fed surrogate model's 5 of 12 shows the read-out can steer sometimes.
  Learning rate, noise and the number of episodes were fixed in advance and not tuned. One training seed per arm.
- Supports the earlier picture: the label-trained read-out (9 of 12) remains the only one that reliably steers; reward-only training of an
  external read-out of projection neurons is much less efficient and was not reliably successful here.
