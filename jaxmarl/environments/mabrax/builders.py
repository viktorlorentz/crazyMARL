import mujoco
from brax.io import mjcf

def make_brax_system(
    num_quads: int,
    cable_length: float,
    payload_mass: float,
    policy_freq: float,
    sim_steps_per_action: int
):
    """
    Generate a Brax system from a MuJoCo XML as a MJX backend.
    """
    from .quad_env_builder import QuadEnvGenerator

    gen = QuadEnvGenerator(
        n_quads=num_quads,
        cable_length=cable_length,
        payload_mass=payload_mass
    )
    xml = gen.generate_xml()
    mj_model = mujoco.MjModel.from_xml_string(xml)
    sys = mjcf.load_model(mj_model)

    # set timestep
    dt = (1.0 / policy_freq) / sim_steps_per_action
    sys.mj_model.opt.timestep = dt
    return sys
