"""Capture actual full-brain rates during ten seconds of embodied frozen execution."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.controller import FrozenPopulation
from noesis_gpu.run import file_digest, source_digest

root = Path(__file__).resolve().parent
project = root.parent
output = root / "outputs/neural-video-10s-20261023"
source = Path("C:/Users/heyma/.codex/worktrees/brain-body-feedback/Project Noesis/src")
sys.path.insert(0, str(source))
from noesis.embodiment import load_training_brain

state = json.loads((root / "continuation-status.json").read_text(encoding="utf-8"))
checkpoint = Path(state["latest_experimental_training_checkpoint"])
assert file_digest(checkpoint) == state["latest_experimental_training_sha256"]
output.mkdir(exist_ok=False)
report = {"status": "initializing", "seconds": 10., "fps": 30, "seed": 10007,
          "backend": "cpu", "learning": False, "motor_exploration": 0,
          "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
          "noesis_source_sha256": source_digest(source), "script_sha256": file_digest(__file__),
          "activity": "Actual per-neuron RateTracker output; exponential smoothing tau=50ms",
          "layout": "Existing schematic anatomical BrainVolume; not measured 3D neuron coordinates",
          "displayed_connections": 1800, "brightest_glowing_neurons_limit": 2000,
          "regional_rates": "Mean over every neuron assigned to each disjoint region group",
          "external_visual_and_odor_inputs": False, "proprioception_and_balance": True,
          "model_promoted": False, "completed_chunks": 0}
started = time.perf_counter()
trace = []
writer = None
try:
    body = FlyGymBatch(1, "cpu")
    cn, _ = load_training_brain(project / "data", output / "brain-cache")
    population = FrozenPopulation(cn, body, seed=10007, checkpoint=checkpoint)
    brain = population.brains[0]
    rates = np.lib.format.open_memmap(output / "rates_hz.npy", mode="w+", dtype=np.float32,
                                     shape=(300, cn.n))
    rates[0] = brain.tracker.rate
    frame_times = [0.]
    observation = body.observe()
    frame_index = 1
    for chunk in range(2000):
        if time.perf_counter() - started > 1800:
            raise TimeoutError("Ten-second neural capture exceeded wall budget")
        commands, pads = population.step(observation, 5.)
        body.advance(commands, pads, 50)
        observation = body.observe()
        trace.append({key: value.copy() for key, value in observation.items()})
        t = (chunk + 1) * .005
        if frame_index < 300 and t + 1e-9 >= frame_index / 30:
            rates[frame_index] = brain.tracker.rate
            frame_times.append(t)
            frame_index += 1
        report["completed_chunks"] = chunk + 1
        if (chunk + 1) % 200 == 0:
            print(json.dumps({"event": "neural_capture", "seconds": t,
                              "frames": frame_index, "spikes": brain.total_spikes}), flush=True)
    assert frame_index == 300 and brain.weights_unchanged()
    rates.flush()
    np.save(output / "frame-times.npy", frame_times)
    cn.neurons[["root_id", "Top in/out region", "Soma side", "Class"]].to_parquet(
        output / "neurons.parquet", index=False)
    report.update(status="capture_complete", neurons=cn.n, edges=cn.W.nnz,
                  simulated_seconds=10., weights_unchanged=True, total_spikes=brain.total_spikes)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print('Neural capture complete; rendering the existing brain map.', flush=True)

    os.environ["MPLBACKEND"] = "Agg"
    spec = importlib.util.spec_from_file_location("recording_brain_map", project / "src/noesis/viz/brain_map.py")
    brain_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(brain_map)
    positions = brain_map.compact_anatomy_positions(cn.neurons)
    pre, post, sign = brain_map.strongest_synapses(cn.W, limit=1800)
    regions = cn.neurons["Top in/out region"].fillna("NO_CONS").str.split(".").str[0].str.upper()
    view = brain_map.BrainVolume(positions, pre, post, sign, anatomical_context=True,
                                context_nodes=np.flatnonzero(~regions.eq("NO_CONS")))
    view.fig.set_size_inches(7.0, 9.0)
    view.fig.set_dpi(100)
    region_sets = [
        ("Optic lobes", {"ME", "LO", "LOP", "AME"}),
        ("Antennal lobes", {"AL"}),
        ("Mushroom bodies", {"MB_CA", "MB_PED", "MB_VL", "MB_ML"}),
        ("Lateral horns", {"LH"}),
        ("Central complex", {"FB", "EB", "PB", "NO"}),
        ("Lateral brain", {"AVLP", "PVLP", "PLP", "WED", "AMMC", "AOTU", "LAL", "VES", "SPS", "IPS"}),
        ("Superior / central brain", {"SMP", "SLP", "SIP", "SCL", "ICL", "IB", "ATL", "CRE", "BU", "GA", "GOR", "EPA"}),
        ("Suboesophageal zone", {"GNG", "PRW", "FLA", "CAN", "SAD"}),
    ]
    for label, tokens in (("Front leg cord (T1)", ("T1", "NTCT")),
                          ("Middle leg cord (T2)", ("T2", "WTCT")),
                          ("Hind leg cord (T3)", ("T3", "HTCT"))):
        region_sets.append((label, {r for r in regions.unique() if any(s in r for s in tokens)}))
    region_sets.append(("No recorded synapses", {"NO_CONS"}))
    masks, used = [], np.zeros(cn.n, bool)
    for label, names in region_sets:
        mask = regions.isin(names).to_numpy() & ~used
        used |= mask
        masks.append((label, mask))
    masks.append(("Other brain / nerve cord", ~used))
    assert np.all(np.sum([mask for _, mask in masks], axis=0) == 1)
    group_rates = np.array([[float(frame[mask].mean()) if mask.any() else 0.
                            for _, mask in masks] for frame in rates])
    np.save(output / "region-mean-rates.npy", group_rates)
    (output / "region-groups.json").write_text(json.dumps([
        {"label": label, "count": int(mask.sum()), "regions": sorted(regions[mask].unique())}
        for label, mask in masks], indent=2), encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
    writer = imageio.get_writer(str(output / "brain.partial.mp4"), fps=30, codec="libx264",
                               quality=8, pixelformat="yuv420p", macro_block_size=1)
    for i, (frame, t) in enumerate(zip(rates, frame_times, strict=True)):
        view.update(frame, t)
        view.fig.canvas.draw()
        left = Image.fromarray(np.asarray(view.fig.canvas.buffer_rgba())[..., :3])
        canvas = Image.new("RGB", (1280, 960), (11, 19, 18))
        canvas.paste(left, (0, 50))
        draw = ImageDraw.Draw(canvas)
        draw.text((710, 35), "BRAIN REGIONS", font=title_font, fill=(224, 236, 231))
        draw.text((710, 79), f"t = {t:05.2f} s  |  mean firing rate (Hz)", font=font, fill=(178, 211, 200))
        for j, (label, mask) in enumerate(masks):
            y = 131 + 48 * j
            mean_rate = group_rates[i, j]
            draw.text((710, y), label, font=small, fill=(209, 222, 217))
            draw.text((1155, y), f"{mean_rate:5.1f}", font=small, fill=(255, 222, 154))
            draw.rectangle((710, y+25, 1235, y+30), fill=(32, 49, 44))
            width = 525 * min(mean_rate / 100., 1.)
            draw.rectangle((710, y+25, 710+width, y+30), fill=(227, 197, 124))
        draw.text((710, 790), "Glow: measured firing, smoothed over 50 ms", font=small, fill=(227, 211, 166))
        draw.text((710, 819), "Schematic positions; 1,800 sampled real links", font=small, fill=(148, 173, 164))
        draw.text((710, 848), "Joint / load / balance feedback connected", font=small, fill=(148, 173, 164))
        draw.text((710, 877), "External vision and odor inputs not connected", font=small, fill=(148, 173, 164))
        draw.text((710, 906), "No learning or exploration | real-time playback", font=small, fill=(148, 173, 164))
        writer.append_data(np.asarray(canvas))
        if i in (0, 75, 150, 299):
            canvas.save(output / f"frame-{i:03d}.png")
        if (i+1) % 60 == 0:
            print(json.dumps({"event": "render", "frames": i+1}), flush=True)
    writer.close()
    writer = None
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(output / "brain.partial.mp4"))
    assert count == 300 and abs(duration - 10.) < .05
    assert source_digest(source) == report["noesis_source_sha256"]
    assert file_digest(checkpoint) == report["checkpoint_sha256"]
    (output / "brain.partial.mp4").rename(output / "brain-activity-10-seconds.mp4")
    report.update(status="completed", encoded_frames=count, encoded_duration=duration,
                  video_sha256=file_digest(output / "brain-activity-10-seconds.mp4"),
                  visualization_source_sha256=file_digest(project / "src/noesis/viz/brain_map.py"))
except BaseException as error:
    report.update(status="incomplete", error=str(error))
    raise
finally:
    if writer is not None:
        writer.close()
    if trace:
        np.savez_compressed(output / "observations.npz", **{
            key: np.stack([sample[key] for sample in trace]) for key in trace[0]})
    report["wall_seconds"] = time.perf_counter() - started
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
