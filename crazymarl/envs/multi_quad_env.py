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
from crazymarl.utils.multi_quad_utils import R_from_quat, angle_between

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

    def step(self, state: State, action: jp.ndarray) -> State:
        cfg = self.cfg
        prev_act = state.obs[-self.sys.nu:]
        max_thrust = state.metrics['max_thrust']
        cmds = jp.clip(0.5*(action+1.0),0.0,1.0) * max_thrust
        ps = self.pipeline_step(state.pipeline_state, cmds)

        nk = jax.random.PRNGKey(0)
        nk = jax.random.fold_in(nk, jp.int32(ps.time*1e6))
        nk = jax.random.fold_in(nk, jp.int32(jp.sum(ps.xpos)*1e3))
        nk = jax.random.fold_in(nk, jp.int32(jp.sum(ps.cvel)*1e3))
        if cfg.act_noise:
            noise = jax.random.normal(nk, action.shape)
            cmds = cmds + cfg.act_noise * max_thrust * noise

        angles=[]; qp=[]
        up = jp.array([0.0,0.0,1.0])
        for b in self.ids["quad_body_ids"]:
            quat = ps.xquat[b]
            angles.append(angle_between(R_from_quat(quat)[:,2], up))
            qp.append(ps.xpos[b])
        angles = jp.stack(angles); qp = jp.stack(qp)

        d = qp[:,None,:] - qp[None,:,:]
        min_d = jp.min(jp.where(jp.eye(cfg.num_quads, dtype=bool), jp.inf, jp.linalg.norm(d, axis=-1)))
        quad_col = min_d < 0.15
        ground_col = jp.any(qp[:,2]<0.03) | (ps.xpos[self.ids["payload_body_id"]][2]<0.03)

        too_tilt = jp.any(jp.abs(angles)>jp.radians(150))
        below_pl = jp.any(qp[:,2] < ps.xpos[self.ids["payload_body_id"]][2]-0.15)
        oob = too_tilt | below_pl

        tgt = self.target_position
        if self.trajectory is not None:
            idx = jp.clip(jp.floor(ps.time/self.time_per_action).astype(jp.int32), 0, self.trajectory.shape[0]-1)
            tgt = self.trajectory[idx]

        obs = build_obs(ps, action, tgt, cfg.obs_noise, nk, self.ids)
        coll = quad_col | ground_col
        reward = calc_reward(obs, ps.time, coll, oob, action, angles, prev_act, tgt, ps, max_thrust, cfg)
        done = (oob | coll).astype(jp.float32)
        metrics = {'time': ps.time, 'reward': reward, 'max_thrust': max_thrust}
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, metrics=metrics)
