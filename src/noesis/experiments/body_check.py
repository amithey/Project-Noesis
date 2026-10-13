"""Physics acceptance checks for the body, with no brain.

1. Stance: with every muscle relaxed the body must stand on its tissue elasticity alone.
2. Range of motion: full agonist (then antagonist) activation of one leg's muscles must bring
   its joints to the extreme and never push any beyond the limit (other legs stand).
3. Floor: the deepest floor penetration seen in every run.
4. Information only: a command 0.3 rad *beyond* the range (the muscle model never issues one),
   to show how hard the joint stop is.
"""

from __future__ import annotations

import argparse

import numpy as np

from noesis.body import BodyParams, MuscleModel, make_fly


def _sim(params: BodyParams):
    from flygym import SingleFlySimulation

    kw = {}
    try:  # the shared physical world, if present
        from noesis import world
        kw = {"arena": world.floor_arena(), "gravity": world.GRAVITY}
    except ImportError:
        world = None
    fly = make_fly(params)
    sim = SingleFlySimulation(fly=fly, cameras=[], timestep=1e-4, **kw)
    if world is not None:
        world.configure_floor_contacts(sim)
    return fly, sim


def _floor_depth_mm(sim) -> float:
    """Deepest penetration (mm, >= 0) of any contact involving a plane geom."""
    import mujoco

    m, d = sim.physics.model.ptr, sim.physics.data.ptr
    if d.ncon == 0:
        return 0.0
    plane = m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    c = d.contact
    on_floor = plane[c.geom1[: d.ncon]] | plane[c.geom2[: d.ncon]]
    return float(max(0.0, -c.dist[: d.ncon][on_floor].min())) if on_floor.any() else 0.0


def _run(sim, command: np.ndarray, seconds: float):
    obs, _ = sim.reset()
    joints = obs["joints"][0]
    deepest, lo, hi = 0.0, joints.copy(), joints.copy()
    for _ in range(round(seconds / sim.timestep)):
        obs, *_ = sim.step({"joints": command, "adhesion": np.ones(6, int)})
        j = obs["joints"][0]
        lo, hi = np.minimum(lo, j), np.maximum(hi, j)
        deepest = max(deepest, _floor_depth_mm(sim))
    return obs, lo, hi, deepest


def main() -> None:
    from flygym import PhysicsError
    from scipy import sparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stand-s", type=float, default=0.6)
    ap.add_argument("--push-s", type=float, default=0.4)
    ap.add_argument("--beyond", action="store_true", help="also run the informational test 4")
    ap.add_argument("--limit-solref", type=float, default=BodyParams.limit_solref)
    ap.add_argument("--legs", default="LF,RM,LH", help="legs to test one at a time")
    ap.add_argument("--damping", type=float, default=BodyParams.joint_damping)
    ap.add_argument("--tolerance", type=float, default=0.05, help="rad beyond limit allowed")
    a = ap.parse_args()

    params = BodyParams(limit_solref=a.limit_solref, joint_damping=a.damping)
    fly, sim = _sim(params)
    names = list(fly.actuated_joints)
    n = len(names)
    zero = sparse.csr_matrix((n, 1))
    muscles = MuscleModel(zero, zero, names, params)  # only for pose/limit arrays
    ok = True

    print("== 1. stance (muscles relaxed)")
    obs, _, _, depth = _run(sim, muscles.stand, a.stand_s)
    err = np.abs(obs["joints"][0] - muscles.stand).max()
    z, up = obs["fly"][0][2], obs["cardinal_vectors"][2][2]
    passed = up > 0.9 and 0.9 < z < 2.0
    ok &= passed
    print(f"   body height {z:.2f} mm, uprightness {up:+.2f}, max joint error {err:.2f} rad, "
          f"deepest floor penetration {depth * 1000:.1f} um -> {'PASS' if passed else 'FAIL'}")

    # 2. one leg at a time at the extreme of its range, the others standing: isolates the joint stop
    for leg in a.legs.split(","):
        mine = np.array([f"joint_{leg}" in nm and nm.startswith(f"joint_{leg}") for nm in names])
        for kind, ext in (("hi", muscles.hi_eq), ("lo", muscles.lo_eq)):
            cmd = np.where(mine, ext, muscles.stand)
            label = f"{leg} leg, full {'agonist' if kind == 'hi' else 'antagonist'} activation"
            try:
                _, lo, hi, depth = _run(sim, cmd, a.push_s)
            except PhysicsError as err_:
                print(f"== 2. {label}: physics became unstable ({str(err_)[-40:]}) -> FAIL")
                ok = False
                continue
            over = (np.maximum(hi - muscles.hi, 0) if kind == "hi" else np.maximum(muscles.lo - lo, 0))[mine]
            got = (hi >= muscles.hi_eq - 0.1) if kind == "hi" else (lo <= muscles.lo_eq + 0.1)
            reached = int(got[mine].sum())
            passed = over.max() <= a.tolerance and reached >= 0.8 * mine.sum()
            ok &= passed
            print(f"== 2. {label}: worst overshoot {over.max():.3f} rad, {reached}/{int(mine.sum())} joints "
                  f"reached the extreme, floor penetration {depth * 1000:.1f} um -> {'PASS' if passed else 'FAIL'}")

    # information only: every joint of every leg at once (whole body is flung around)
    for kind, ext in (("hi", muscles.hi_eq), ("lo", muscles.lo_eq)):
        try:
            _, lo, hi, _ = _run(sim, ext, a.push_s)
            over = np.maximum(hi - muscles.hi, 0) if kind == "hi" else np.maximum(muscles.lo - lo, 0)
            print(f"== info: all {n} joints at once, {kind}: worst overshoot {over.max():.3f} rad")
        except PhysicsError:
            print(f"== info: all {n} joints at once, {kind}: physics became unstable")

    if a.beyond:
        print("== 4. information: command 0.3 rad beyond the range (never issued by the muscle model)")
        try:
            _, lo, hi, _ = _run(sim, muscles.hi + 0.3, a.push_s)
            print(f"   worst overshoot {np.maximum(hi - muscles.hi, 0).max():.3f} rad")
        except PhysicsError:
            print("   physics became unstable")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
