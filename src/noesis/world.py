"""Shared physical world: infinite floor at z=0, gravity in millimetres/s²."""

GRAVITY = (0.0, 0.0, -9810.0)
FLOOR_COLLISIONS = "all"
# Contact recovery time is two physics steps; critically damped impacts.
# FlyGym's default damping ratio of 1000 is too compliant for body-first falls.
CONTACT_SOLREF = (2e-4, 1.0)
FLOOR_CONTACT_MARGIN_MM = 0.05


def configure_floor_contacts(sim):
    """Start resolving contact just above the floor to absorb impact discretization.

    A 50 micrometre collision skin prevents high-speed body-first impacts from
    stepping through the visible plane. This changes contact forces, not positions.
    """
    import mujoco

    model = sim.physics.model.ptr
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground < 0:
        raise ValueError("physical world has no ground geometry")
    pairs = (model.pair_geom1 == ground) | (model.pair_geom2 == ground)
    model.pair_margin[pairs] = FLOOR_CONTACT_MARGIN_MM


def floor_arena():
    """Solid horizontal plane; every fly geometry must enable floor contacts.

    MuJoCo resolves contacts numerically, with small solver penetration tolerances.
    The visible extent of the plane does not limit its collision extent.
    """
    from flygym.arena import FlatTerrain

    arena = FlatTerrain()
    arena.root_element.find("geom", "ground").pos = (0, 0, 0)
    # Remove the below-floor mirror image, so the floor boundary is unambiguous.
    arena.root_element.find("material", "grid").reflectance = 0
    return arena


def launch_viewer(sim, enriched=False):
    """Open a passive MuJoCo viewer after resetting offscreen GLFW window hints.

    dm_control's GLFWContext sets DOUBLEBUFFER=0 and VISIBLE=0 globally on the
    main thread when creating offscreen cameras (e.g. fly vision). Resetting
    default window hints ensures the interactive viewer window has a backbuffer
    and swaps frames onto the screen on Windows.
    """
    import glfw
    import mujoco.viewer

    if glfw.init():
        glfw.default_window_hints()
    viewer = mujoco.viewer.launch_passive(sim.physics.model.ptr, sim.physics.data.ptr)
    if enriched and hasattr(viewer, "cam"):
        viewer.cam.lookat[:] = (0.5, 0.0, 0.7)
        viewer.cam.distance, viewer.cam.elevation = 22, -50
    viewer.sync()
    return viewer


def enable_wing_and_head_joints(fly) -> None:
    """Attach wing hinge joints and position actuators for the head and wings when using real MJCF.

    No-op on lightweight test doubles whose ``fly.model`` has no ``find`` method.
    """
    if not hasattr(getattr(fly, "model", None), "find"):
        return
    import numpy as np

    wing_joints = []
    for side in ("L", "R"):
        wb = fly.model.find("body", f"{side}Wing")
        if wb is None:
            continue
        for suffix, axis in (("_yaw", (1, 0, 0)), ("", (0, 1, 0)), ("_roll", (0, 0, 1))):
            jname = f"joint_{side}Wing{suffix}"
            if fly.model.find("joint", jname) is None:
                wb.add(
                    "joint", name=jname, dclass="nmf", type="hinge",
                    pos=(0, 0, 0), axis=axis, stiffness=0.05, springref=0.0,
                    damping=0.02, limited="true", range=(-1.2, 1.2),
                )
            wing_joints.append(jname)
    extra_joints = ["joint_Head_yaw", "joint_Head", "joint_Head_roll"] + wing_joints
    for jname in extra_joints:
        if jname in fly.actuated_joints:
            continue
        j = fly.model.find("joint", jname)
        if j is None:
            continue
        j.stiffness = 0.05
        j.damping = 0.02
        act = fly.model.actuator.add(
            "position", name=f"actuator_position_{jname}", joint=jname,
            kp=15.0, ctrlrange="-1000000 1000000", forcerange=(-20, 20), forcelimited=True,
        )
        fly.actuators.append(act)
        fly.actuated_joints.append(jname)
        if jname not in fly.monitored_joints:
            fly.monitored_joints.append(jname)
            s_pos = fly.model.sensor.add("jointpos", name=f"jointpos_{jname}", joint=jname)
            s_vel = fly.model.sensor.add("jointvel", name=f"jointvel_{jname}", joint=jname)
            s_frc = fly.model.sensor.add(
                "actuatorfrc", name=f"actuatorfrc_position_{jname}",
                actuator=f"actuator_position_{jname}",
            )
            fly._actuated_joint_sensors.extend([s_pos, s_vel, s_frc])
            fly._monitored_joint_order.append(len(fly._actuated_joint_sensors) // 3 - 1)
    if hasattr(fly, "_define_action_space"):
        fly.action_space = fly._define_action_space(np.pi)


