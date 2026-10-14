# The fly's sensory habitat

Run `./Start-Fly.ps1 -Video` for a live viewer and a recording. Add `-Learn` to
enable the experimental plasticity rule. The default arena is now `habitat`.
Every control step samples the fly's surroundings and drives annotated sensory
neurons. World objects do not send commands to motor neurons.

## What is in the world

| Feature | Visible cue | Input |
| --- | --- | --- |
| Fruit / sugar patches & drop | Yellow-orange disks and fruit | Fruity, yeast, fermentation and decaying-fruit odor channels; sugar/nutritive taste on contact |
| Water patch & drop | Blue disks | Water taste on contact; humid region |
| Bitter patch & drop | Purple disks | Aversive odor channel; bitter/aversive taste on contact |
| Plant | Green stem and disk | Plant odor channel |
| Warm region & thermal wave | Orange-red disk + traveling wave | Local temperature increases and oscillates |
| Cool region & thermal wave | Pale blue disk + traveling wave | Local temperature decreases and oscillates |
| Dry region & moisture wave | Tan disk + traveling wave | Local humidity and evaporation oscillate |
| Airflow / breeze wave | Travelling spatial-temporal field | Drives antennal, eye and wing mechanosensory bristles/campaniforms |
| Obstacle & floor touch | Brown block and solid floor | Physical collision, spatially attenuated sound envelope, and organ-specific tactile bristles |
| Landmarks | Blue, yellow and green wall panels | Rendered colour contrast in the two eyes |
| Moving dark target | Dark sphere overhead | Autonomous motion/approach in the rendered scene |

An 18 mm-wide enclosure has solid walls, floor, obstacles, and nearby foraging
micro-patches within reach of the fly's legs near the spawn position. The previous
gravity and whole-body floor-contact configuration is preserved. The 2D swirling
odor plume, lighting, moving target, thermal/moisture waves, airflow and sound
envelope change continuously with simulated time. No user needs to manually
stimulate neuron IDs.

## Sensory interfaces

- **Odor:** smooth bounded spatial fields with 2D swirling plumes, sampled by
  FlyGym at the actual left and right antennal and maxillary-palp sensor
  positions. Six channels map to exact `Function` labels in the downloaded
  neuron table. Inputs differ across sides and change when the fly moves.
- **Vision:** FlyGym renders the scene from both eye cameras, including physical
  occlusion, and produces 721 ommatidial readings per eye. R7-labelled neurons
  receive the blue channel and R8-labelled neurons the green channel. Retinal
  locations are assigned reproducibly; `--sensory-map-seed` defaults to 0 and
  remains independent of the episode/network seed. Refresh defaults to 20 Hz.
- **Taste:** only force-bearing MuJoCo contacts involving a leg segment
  (`Tarsus`, `Tibia`, `Femur`) or proboscis (`Haustellum`, `Rostrum`) can
  activate taste. The contacted material/patch determines sugar (including
  labellar fatty/amino acid pegs), water or bitter (including labellar aversive)
  input. Flying over food or touching it with the thorax does not provide taste.
  Internal/pharyngeal and wing taste are not mapped.
- **Heat / humidity:** local fields and traveling waves at the head drive all
  annotated heating, cooling (`Ir21a` and `Ir40a`), humid (`Ir68a`) and
  dry/evaporation (`Ir40a` dry and `Ir68a` evaporation) receptor populations.
- **Sound:** a local periodic amplitude envelope drives Johnston's organ A/B
  afferents. It is not an acoustic wave or pitch-recognition model.
- **Wind / airflow:** a spatial-temporal airflow envelope drives antennal,
  interommatidial, postorbital and wing mechanosensory bristle and campaniform
  afferents.
- **Tactile body / mouth / leg touch:** force-bearing external MuJoCo contacts
  on the thorax, abdomen, mouth/head and individual legs drive the corresponding
  annotated tactile bristle populations.
- Existing leg load, touch and proprioceptive feedback (`SensoryMap`, 3,921
  neurons) and gravity/haltere balance feedback (`BalanceMap`, 1,053 neurons)
  operate in both `newborn` and `learning_lab`.

The current downloaded table maps 1,842 visual, 1,944 odor, 612 taste, 47 thermal
(6 heating, 41 cooling), 71 humidity/evaporation (20 humid, 51 dry/evaporation),
510 sound, 1,727 wind and 1,403 tactile-contact afferent entries: **8,156
environmental input neurons** (and **13,130 total sensory neurons** together with
leg proprioception and balance). Unlabelled and unsupported receptor types are
not silently assigned stimuli.

## Controls and recording

```powershell
# Default habitat, frozen weights, live view and video
.\Start-Fly.ps1 -Video
# Same visible world with its additional brain input disabled
.\Start-Fly.ps1 -NoEnvironmentSenses
# Original flat arena
.\Start-Fly.ps1 -Arena empty
```

Direct CLI options include `--arena habitat|empty`, `--no-environment-senses`,
`--environment-weight` (100 synapse-equivalents by default), `--vision-hz`, and
`--sensory-map-seed`. `--no-senses` disables leg feedback only. For the original
newborn experiment, use `python -m noesis.experiments.newborn --arena habitat`.

The learning laboratory saves:

- A video of the whole habitat when `--video` is enabled.
- `episode_NNN.eyes.png`: the most recent binocular ommatidial view, displayed
  as green/blue RGB for inspection, not a human-like view.
- `episode_NNN.senses.npz`: input neuron IDs, most recent drive rates, retina,
  odor readings and physical taste contacts. Rates are from the last control
  input; retina/odor/contact arrays are the final physical observations.
- `episode_NNN.sensory-map.json`: counts of mapped neurons by modality.
- CSV columns for experienced odor, retinal intensity, taste contacts,
  temperature, humidity, sound envelope and stimulated input counts. Sensory
  telemetry describes the input at the beginning of each control interval.
- `config.json`: arena layout, source fields, gains, versions and seeds.

## Interpretation and validation

These sensory response curves and environmental fields are **model assumptions**.
The table supplies receptor labels, not measured odor affinities, firing-rate
calibration, retinotopy, or a full ultraviolet spectral model. R7/R8 channel
assignments are an exploratory approximation. Heat/humidity fields do not solve
heat transfer or evaporation, and water is a surface patch rather than a fluid.
The dark sphere is a visual stimulus, not a model of a predator's behavior.

Tests verify distance/time-dependent fields, organ-specific routing, no taste
without contact, colour changes reaching actual rendered retina, moving actual
antennae changing perceived odor, and real contact forces activating tarsal or
mouth taste only at food. The full-connectome 0.6 s recorded smoke run completed
with the sensory habitat. This establishes the working input loop, not learned
food seeking, escape behavior, coordinated walking or flight.

The local recording is `outputs/habitat-preview-20261013/episode_000.mp4`.
