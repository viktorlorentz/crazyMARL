import jax
from jax import numpy as jp
import mujoco
from brax import base
from brax.envs.base import PipelineEnv, State
from crazymarl.configs.multi_quad_config import MultiQuadConfig
from crazymarl.builders.quad_env_builder import make_brax_system, get_body_and_joint_ids
from crazymarl.utils.multi_quad_sampling import generate_filtered_configuration_batch
from crazymarl.observations.multi_quad_observation import build_obs
from crazymarl.rewards.multi_quad_reward import calc_reward
from crazymarl.utils.multi_quad_utils import R_from_quat, upright_angles

class MultiQuadEnv(PipelineEnv):
    def __init__(self, **kwargs):
        # Remove framework-specific kwargs not used by config
        for _k in ('backend', 'n_frames', 'episode_length'):
            kwargs.pop(_k, None)
        cfg = MultiQuadConfig(**kwargs)
        sys = make_brax_system(
            cfg.num_quads,
            cfg.cable_length,
            cfg.payload_mass,
            cfg.policy_freq,
            cfg.sim_steps_per_action
        )

        self.num_quads = cfg.num_quads

        super().__init__(sys, backend='mjx', n_frames=cfg.sim_steps_per_action)
        self.cfg = cfg
       
        self.time_per_action = 1.0 / cfg.policy_freq
        self.base_max_thrust = 0.14
        self.goal_center = jp.array([0.0, 0.0, 1.5])
        self.target_position = self.goal_center
        self.trajectory = None
        if cfg.trajectory is not None:
            traj = jp.array(cfg.trajectory, dtype=jp.float32)
            self.trajectory = traj.reshape(-1,3)
            self.target_position = self.trajectory[0]
        self.ids = get_body_and_joint_ids(sys, num_quads=self.num_quads)


    def reset(self, rng: jax.Array) -> State:
        cfg = self.cfg
        rng, mt_rng = jax.random.split(rng)
        factor = jax.random.uniform(mt_rng, (), minval=1.0-cfg.max_thrust_range, maxval=1.0)
        max_thrust = self.base_max_thrust * factor

        rng, r1, r2, rc = jax.random.split(rng, 4)
        base_qpos = self.sys.qpos0
        qvel = jp.zeros(self.sys.nv)
        ang_std = 20*jp.pi/180; lin_std = 0.2
        for b in self.ids["quad_body_ids"]:
            lin = jax.random.normal(r2,(3,))*lin_std
            ang = jax.random.normal(r2,(3,))*ang_std
            i = b*6
            qvel = qvel.at[i:i+3].set(lin)
            qvel = qvel.at[i+3:i+6].set(ang)

        payload_pos, quad_pos = generate_filtered_configuration_batch(
            rc, 1, cfg.num_quads, cfg.cable_length, self.target_position, cfg.target_start_ratio
        )
        payload_pos = payload_pos[0]
        quad_pos = quad_pos[0]

        # orientations
        rng, re = jax.random.split(rng)
        keys = jax.random.split(re, cfg.num_quads*3)
        std, clip = 10*jp.pi/180, 60*jp.pi/180
        quats=[]
        for i in range(cfg.num_quads):
            k0,k1,k2 = keys[3*i:3*i+3]
            roll = jp.clip(jax.random.normal(k0)*std, -clip, clip)
            pitch= jp.clip(jax.random.normal(k1)*std, -clip, clip)
            yaw  = jax.random.uniform(k2, minval=-jp.pi, maxval=jp.pi)
            cond = quad_pos[i, 2] < 0.02
            roll  = jp.where(cond, 0.0, roll)
            pitch = jp.where(cond, 0.0, pitch)
            cr, sr = jp.cos(roll*0.5), jp.sin(roll*0.5)
            cp, sp = jp.cos(pitch*0.5), jp.sin(pitch*0.5)
            cy, sy = jp.cos(yaw*0.5), jp.sin(yaw*0.5)
            quats.append(jp.array([
                cr*cp*cy + sr*sp*sy,
                sr*cp*cy - cr*sp*sy,
                cr*sp*cy + sr*cp*sy,
                cr*cp*sy - sr*sp*cy
            ]))
        quats = jp.stack(quats)

        qpos = base_qpos
        qpos = qpos.at[self.ids["payload_qpos_start"]:self.ids["payload_qpos_start"]+3].set(payload_pos)
        for i,s in enumerate(self.ids["quad_qpos_starts"]):
            qpos = qpos.at[s:s+3].set(quad_pos[i])
            qpos = qpos.at[s+3:s+7].set(quats[i])

        ps = self.pipeline_init(qpos, qvel)
        last_act = jp.zeros(self.sys.nu)
        rng, nk = jax.random.split(rng)
        obs = build_obs(ps, last_act, self.target_position, cfg.obs_noise, nk, self.ids)
        return State(ps, obs, jp.array(0.0), jp.array(0.0), {'time': ps.time, 'reward': 0.0, 'max_thrust': max_thrust})

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self.cfg
   
       
        # Scale actions from [-1, 1] to thrust commands in [0, max_thrust].
        max_thrust = state.metrics['max_thrust']
        thrust_cmds = 0.5 * (action + 1.0)
        thrust_cmds = jp.clip(thrust_cmds, 0.0, 1.0)
        action_scaled = thrust_cmds * max_thrust

 
        ps = self.pipeline_step(state.pipeline_state, action_scaled)

        # Generate a dynamic noise_key using pipeline_state fields.
        noise_key = jax.random.PRNGKey(0)
        noise_key = jax.random.fold_in(noise_key, jp.int32(ps.time * 1e6))
        noise_key = jax.random.fold_in(noise_key, jp.int32(jp.sum(ps.xpos) * 1e3))
        noise_key = jax.random.fold_in(noise_key, jp.int32(jp.sum(ps.cvel) * 1e3))

        # Add actuator noise.
        if cfg.act_noise:
            noise = jax.random.normal(noise_key, shape=action_scaled.shape)
            action_scaled = action_scaled + cfg.act_noise * max_thrust * noise

        quad_body_ids = jp.array(self.ids["quad_body_ids"])

        up = jp.array([0.0, 0.0, 1.0])
        # collect orientations & positions
        quats = ps.xquat[quad_body_ids]  # (num_quads, 4)
        angles = upright_angles(R_from_quat(quats))  # (num_quads,)

        qp = ps.xpos[quad_body_ids] # (num_quads, 3)

        # pairwise quad-quad collision TODO: make this use proper mjx collision detection
        dists = jp.linalg.norm(qp[:, None, :] - qp[None, :, :], axis=-1)
        eye  = jp.eye(cfg.num_quads, dtype=bool)
        min_dist = jp.min(jp.where(eye, jp.inf, dists))
        quad_collision = min_dist < 0.15

        # ground collision if any quads AND payload near ground
        ground_collision_quad    = jp.any(qp[:, 2] < 0.03)
        ground_collision_payload = ps.xpos[self.ids["payload_body_id"]][2] < 0.03
        ground_collision = jp.logical_or(ground_collision_quad, ground_collision_payload)
        collision       = jp.logical_or(quad_collision, ground_collision)

        # out-of-bounds if any quad tilts too far or goes under payload
        too_tilted = jp.any(jp.abs(angles) > jp.radians(150))
        below_pl   = jp.any(qp[:, 2] < ps.xpos[self.ids["payload_body_id"]][2] - 0.15)
        out_of_bounds = jp.logical_or(too_tilted, below_pl)

        # out of bounds for pos error shrinking with time
        # payload_pos = ps.xpos[self.payload_body_id]
        # payload_error = self.target_position - payload_pos
        # payload_error_norm = jp.linalg.norm(payload_error)
        # max_time_to_target = self.max_time * 0.75
        # time_progress = jp.clip(ps.time / max_time_to_target, 0.0, 1.0)
        # max_payload_error = 4 * (1 - time_progress) + 0.05 # allow for 5cm error at the target
        # out_of_bounds = jp.logical_or(out_of_bounds, payload_error_norm > max_payload_error)


        # set target if trajectory is provided
        target_position = self.target_position
        if self.trajectory is not None and self.trajectory.shape[0] > 0:
            # get the next target position from the trajectory
            target_idx = jp.clip(
                jp.floor(ps.time  / self.time_per_action).astype(jp.int32),
                0, self.trajectory.shape[0] - 1
            ) 
            target_position = self.trajectory[target_idx]



        obs = build_obs(ps, action, target_position, cfg.obs_noise, noise_key, self.ids)
        reward = calc_reward(obs, ps.time, collision, out_of_bounds, action, angles, target_position, ps, max_thrust, cfg)

        # dont terminate ground collision on ground start
        ground_collision = jp.logical_and(
        ground_collision,
        jp.logical_or(
            ps.time > 3, # allow 2 seconds for takeoff
            ps.cvel[self.ids["payload_body_id"]][2] < -3.0,
        )
        )

        collision = jp.logical_or(quad_collision, ground_collision)
        
        done = jp.logical_or(out_of_bounds, collision)
    
        
        done = done * 1.0

        metrics = {
        'time': ps.time,
        'reward': reward,
        'max_thrust': state.metrics['max_thrust']
        }
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, metrics=metrics)

