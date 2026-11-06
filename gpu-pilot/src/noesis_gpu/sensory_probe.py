"""Isolated frozen thermosensory transport probe; no training, motor policy or shared activation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HELD_OUT = frozenset((10,11,200,201,202))
MODES = ("plain","off","zero","constant","active")

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=MODES,required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--core",type=Path,required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(argv)
    if args.seed in HELD_OUT:
        ap.error("Final validation seeds are prohibited")
    sys.path.insert(0,str(args.core.resolve()))
    from . import controller, run
    from .sensed_brain import GradientField, sensed_class
    from .senses_bridge import select_modality

    original=controller.FrozenBrain
    chosen=original if args.mode=="plain" else sensed_class(original)
    created=[]
    class ObservedBrain(chosen):
        def __init__(self,connectome,rest,lo,hi,**options):
            if args.mode!="plain":
                options.update(field=GradientField(axis=0,origin=-1.,length=2.),gain=50.,
                               mode=args.mode,level=.5)
            super().__init__(connectome,rest,lo,hi,**options)
            taken=np.union1d(self.senses.idx,self.balance.idx)
            self.warm=select_modality(connectome.neurons,"thermo_warm",taken=taken)
            self.cool=select_modality(connectome.neurons,"thermo_cool",taken=np.union1d(taken,self.warm))
            if len(self.warm)!=6 or len(self.cool)!=6:
                raise ValueError("Expected exact six warm/six cool afferents in the transferred dataset")
            self.recorded={k:[] for k in ("u","warm_requested_hz","cool_requested_hz","input_position",
                                          "warm_spikes","cool_spikes","total_spikes","commands","adhesion")}
            created.append(self)
        def _extra_drive_parts(self,observation):
            parts=super()._extra_drive_parts(observation)
            self._requested_warm=self._requested_cool=0.0
            for idx,rate,weight in parts:
                idx,rate=np.asarray(idx),np.broadcast_to(rate,np.asarray(idx).shape)
                if not np.isin(idx,np.union1d(self.warm,self.cool)).all():
                    raise ValueError("Unexpected extra sensory target")
                self._requested_warm+=float(rate[np.isin(idx,self.warm)].sum()/len(self.warm))
                self._requested_cool+=float(rate[np.isin(idx,self.cool)].sum()/len(self.cool))
            return parts
        def step(self,observation,dt_ms):
            command,pads=super().step(observation,dt_ms)
            self.recorded["u"].append(float(getattr(self,"last_u",0.)))
            self.recorded["warm_requested_hz"].append(self._requested_warm)
            self.recorded["cool_requested_hz"].append(self._requested_cool)
            self.recorded["input_position"].append(np.asarray(observation["position"]).copy())
            self.recorded["warm_spikes"].append(int(np.isin(self.last_spikes,self.warm).sum()))
            self.recorded["cool_spikes"].append(int(np.isin(self.last_spikes,self.cool).sum()))
            self.recorded["total_spikes"].append(int(len(self.last_spikes)))
            self.recorded["commands"].append(command.copy())
            self.recorded["adhesion"].append(pads.copy())
            return command,pads
    command=["--backend","cpu","--worlds","1","--seconds","5","--wall-seconds","300",
             "--seed",str(args.seed),"--noesis-source",str(args.core.resolve()),"--data",str(args.data.resolve()),
             "--checkpoint",str(args.checkpoint.resolve()),"--output",str(args.output)]
    options=run.parser().parse_args(command)
    controller.FrozenBrain=ObservedBrain
    try:
        report=run.run(options)
    finally:
        controller.FrozenBrain=original
    if report["status"]!="completed_transport_pilot" or len(created)!=1:
        raise RuntimeError("Probe incomplete or wrong population size")
    brain=created[0]
    arrays={k:np.asarray(v) for k,v in brain.recorded.items()}
    assert len(arrays["u"])==1000 and all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(args.output/"sensory-observations.npz",**arrays)
    report.update(sensory_mode=args.mode,stimulus_protocol="normalized thermo gradient x,origin=-1,length=2; constant=.5",
                  gain_hz_per_signal_unit=50.,external_thermosensory_input=args.mode in ("constant","active"),
                  warm_afferent_count=6,cool_afferent_count=6,learning=False,
                  observer_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report["senses"]=[*report["senses"],"thermo-mode:"+args.mode]
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","sensory_mode","completed_control_chunks","weights_unchanged")}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
