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

    dm_control's eye renderer sets ``VISIBLE=0`` and ``DOUBLEBUFFER=0`` on the
    process. A viewer created afterwards is a white window that never swaps a
    frame. Reset the hints, then open the window.
    """
    import glfw
    import mujoco.viewer

    if glfw.init():
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, 1)
        glfw.window_hint(glfw.DOUBLEBUFFER, 1)
    viewer = mujoco.viewer.launch_passive(sim.physics.model.ptr, sim.physics.data.ptr)
    if enriched and hasattr(viewer, "cam"):
        viewer.cam.lookat[:] = (0.5, 0.0, 0.7)
        viewer.cam.distance, viewer.cam.elevation = 22, -50
    viewer.sync()
    return viewer
