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
