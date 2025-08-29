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

        if not cfg.payload:
            cfg.payload_mass = 0.0
            cfg.cable_length = 0.0
            if cfg.num_quads > 1:
                raise NotImplementedError("Quad swarms without payload not yet supported. MultiQuadEnv requires payload for multiple quads. Set payload=True in config.")

        sys = make_brax_system(
            cfg.num_quads,
            cfg.cable_length,
            cfg.payload_mass,
            cfg.policy_freq,
            cfg.sim_steps_per_action,
            cfg.payload,
        )

        self.num_quads = cfg.num_quads

        super().__init__(sys, backend='mjx', n_frames=cfg.sim_steps_per_action)
        self.cfg = cfg
       
        self.time_per_action = 1.0 / cfg.policy_freq
        self.dynamic_decay_tau = 0.075 # roughly 0 after 300ms
        self.base_max_thrust = 0.12
        self.goal_center = jp.array([0.0, 0.0, 1.5])
        self.target_position = self.goal_center
        self.trajectory = None
        if cfg.trajectory is not None:
            traj = jp.array(cfg.trajectory, dtype=jp.float32)
            self.trajectory = traj.reshape(-1,3)
            self.target_position = self.trajectory[0]
        self.ids = get_body_and_joint_ids(sys, num_quads=self.num_quads)

        # External disturbance configuration
        self.disturbance_interval_s = 3.0  # average one event every 2 seconds
        self.disturbance_force_range = (0.0, 0.05)    # Newtons
        self.disturbance_torque_range = (0.0, 0.03)   # N·m
        # Bias torque toward yaw (body z-axis in world frame)
        self.torque_yaw_bias_kappa = 6.0   # larger => stronger alignment with yaw axis
        self.torque_yaw_noise_std = 1.0    # noise around yaw axis
        # Add payload disturbance force range (reuses quad range by default)
        self.payload_disturbance_force_range = (0.0, 5.0) # Newtons
        # Random RPM jump config (average one event per second)
        self.rpm_jump_interval_s = 1.0
        self.rpm_jump_std = 0.05  # std of additive jump in filtered RPM proxy units


    def _build_disturbance_xfrc(self, ps_in):
        # Deterministic RNG from current physics state (stateless sampling)
        dk = jax.random.PRNGKey(12345)
        dk = jax.random.fold_in(dk, jp.int32(ps_in.time * 1e6))
        dk = jax.random.fold_in(dk, jp.int32(jp.sum(ps_in.xpos) * 1e3))
        dk = jax.random.fold_in(dk, jp.int32(jp.sum(ps_in.cvel) * 1e3))
        # Extend splits to include payload disturbance keys
        dk, k_evt, k_body, k_fdir, k_fmag, k_tdir, k_tmag, k_evt_pl, k_fdir_pl, k_fmag_pl = jax.random.split(dk, 10)

        # Event probability => expected one every disturbance_interval_s seconds
        p_event = jp.clip(self.time_per_action / self.disturbance_interval_s, 0.0, 1.0)
        event = jax.random.bernoulli(k_evt, p=p_event)
        event_f = event.astype(jp.float32)

        quad_body_ids = jp.array(self.ids["quad_body_ids"])
        body_idx = jax.random.randint(k_body, (), 0, quad_body_ids.shape[0])
        body_id = quad_body_ids[body_idx]

        # Random force direction and magnitude
        f_dir_raw = jax.random.normal(k_fdir, (3,))
        f_dir = f_dir_raw / (jp.linalg.norm(f_dir_raw) + 1e-6)
        f_mag = jax.random.uniform(
            k_fmag, (), minval=self.disturbance_force_range[0], maxval=self.disturbance_force_range[1]
        )
        force_w = f_dir * f_mag

        # Random torque direction and magnitude (biased toward quad's yaw axis)
        # Compute body z-axis (yaw axis) in world frame for the selected quad
        quats = ps_in.xquat[quad_body_ids]            # (num_quads, 4)
        R = R_from_quat(quats)                        # (num_quads, 3, 3)
        body_z_axes = R[:, :, 2]                      # (num_quads, 3)
        body_z_dir = body_z_axes[body_idx]            # (3,)

        # Sample a noisy, biased direction toward body z-axis
        t_noise = jax.random.normal(k_tdir, (3,)) * self.torque_yaw_noise_std
        t_dir_raw = self.torque_yaw_bias_kappa * body_z_dir + t_noise
        t_dir = t_dir_raw / (jp.linalg.norm(t_dir_raw) + 1e-6)

        t_mag = jax.random.uniform(
            k_tmag, (), minval=self.disturbance_torque_range[0], maxval=self.disturbance_torque_range[1]
        )
        torque_w = t_dir * t_mag

        # Build xfrc_applied for a single step (world frame: [fx,fy,fz, tx,ty,tz])
        xfrc_step = jp.zeros_like(ps_in.xfrc_applied)
        xfrc_vec = jp.concatenate([force_w, torque_w]) * event_f
        xfrc_step = xfrc_step.at[body_id].set(xfrc_vec)

        # Payload disturbance
        if self.cfg.payload and "payload_body_id" in self.ids:
            p_event_pl = jp.clip(self.time_per_action / self.disturbance_interval_s, 0.0, 1.0)
            event_pl = jax.random.bernoulli(k_evt_pl, p=p_event_pl).astype(jp.float32)
            f_dir_raw_pl = jax.random.normal(k_fdir_pl, (3,))
            f_dir_pl = f_dir_raw_pl / (jp.linalg.norm(f_dir_raw_pl) + 1e-6)
            f_mag_pl = jax.random.uniform(
                k_fmag_pl, (), 
                minval=self.payload_disturbance_force_range[0],
                maxval=self.payload_disturbance_force_range[1]
            )
            force_pl = f_dir_pl * f_mag_pl * event_pl
            # Add force (first 3 comps) to payload; torque left zero
            xfrc_step = xfrc_step.at[self.ids["payload_body_id"], :3].add(force_pl)
        # Disturbance flag: 1 if any non-zero wrench applied
        disturbance_flag = (jp.sum(jp.abs(xfrc_step)) > 0).astype(jp.float32)
        return xfrc_step, disturbance_flag

    def reset(self, rng: jax.Array) -> State:
        cfg = self.cfg
        rng, mt_rng = jax.random.split(rng)
        motor_offsets = 0.05 * cfg.max_thrust_range * ( jax.random.normal(mt_rng, (self.sys.nu,)))
        max_thrust = jax.random.uniform(mt_rng, minval=0.11, maxval=0.135)
        max_thrust += motor_offsets

        max_thrust = jp.clip(max_thrust, 0.10, 0.14) 

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

        tau = cfg.motor_tau * jp.clip(1 + jax.random.normal(rng, ()) * 0.3, 0.1, 1.5)

        motor_alpha = self.dt / tau

        qpos = base_qpos
        if cfg.payload:
            qpos = qpos.at[self.ids["payload_qpos_start"]:self.ids["payload_qpos_start"]+3].set(payload_pos)
        for i,s in enumerate(self.ids["quad_qpos_starts"]):
            qpos = qpos.at[s:s+3].set(quad_pos[i])
            qpos = qpos.at[s+3:s+7].set(quats[i])

        ps = self.pipeline_init(qpos, qvel)
        last_act = -jp.ones((self.sys.nu,))

        last_filtered_rpm_proxy = jp.clip(0.85+0.1*jax.random.normal(rng, shape=(self.sys.nu,)), 0, 1)
        
        # Zero RPM proxy for any quad whose initial z position is on the ground (< 0.02)
        motors_per_quad = self.sys.nu // self.num_quads
        grounded = quad_pos[:, 2] < 0.02                          # (num_quads,)
        lfrp = last_filtered_rpm_proxy.reshape((self.num_quads, motors_per_quad))
        lfrp = jp.where(grounded[:, None], 0.0, lfrp)
        last_filtered_rpm_proxy = lfrp.reshape((-1,))

        rng, nk = jax.random.split(rng)
        obs = build_obs(ps, last_act, self.target_position, cfg.obs_noise, nk, self.ids, payload=cfg.payload)
        return State(ps, obs, jp.array(0.0), jp.array(0.0), 
                     {'time': ps.time,
                      'reward': 0.0,
                      'max_thrust': max_thrust,
                      'filtered_rpm_proxy': last_filtered_rpm_proxy,
                      'motor_alpha': motor_alpha,
                      'dynamic': 1.1
                        })

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self.cfg
   
       
        # Scale actions from [-1, 1] to thrust commands in [0, max_thrust].
        max_thrust = state.metrics['max_thrust']
        thrust_cmds = 0.5 * (action + 1.0)
        thrust_cmds = jp.clip(thrust_cmds, 0.0, 1.0)
        action_scaled = thrust_cmds * max_thrust


        alpha = state.metrics['motor_alpha']
        rpm_proxy = jp.sqrt(action_scaled)  # we use sqrt of thrust as a proxy for rotor speed
        r_prev = state.metrics['filtered_rpm_proxy']
        filtered_rpm_proxy = r_prev + alpha * (rpm_proxy - r_prev)

        filtered_thrust = jp.square(filtered_rpm_proxy)  # convert back to thrust
        #filtered_thrust = jp.square(rpm_proxy)

        # Build disturbance xfrc for this step
        ps_in = state.pipeline_state
        xfrc_step, disturbance_flag = self._build_disturbance_xfrc(ps_in)
        ps_in = ps_in.replace(xfrc_applied=xfrc_step)

        # Step physics with disturbance
        ps = self.pipeline_step(ps_in, filtered_thrust)


        # Generate a dynamic noise_key using pipeline_state fields.
        noise_key = jax.random.PRNGKey(0)
        noise_key = jax.random.fold_in(noise_key, jp.int32(ps.time * 1e6))
        noise_key = jax.random.fold_in(noise_key, jp.int32(jp.sum(ps.xpos) * 1e3))
        noise_key = jax.random.fold_in(noise_key, jp.int32(jp.sum(ps.cvel) * 1e3))

        # Add actuator noise (action noise; applied to scaled action but not re-stepped here)
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
        ground_collision_quad    = jp.any(qp[:, 2] < 0.05)
        if cfg.payload:
            ground_collision_payload = ps.xpos[self.ids["payload_body_id"]][2] < 0.03
            ground_collision = jp.logical_or(ground_collision_quad, ground_collision_payload)
        else:
            ground_collision = ground_collision_quad
        collision       = jp.logical_or(quad_collision, ground_collision)

        # out-of-bounds if any quad tilts too far or goes under payload
        too_tilted = jp.any(jp.abs(angles) > jp.radians(170))
        if cfg.payload:
            below_pl   = jp.any(qp[:, 2] < ps.xpos[self.ids["payload_body_id"]][2] - 0.15)
            out_of_bounds = jp.logical_or(too_tilted, below_pl)
        else:
            out_of_bounds = False

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



        obs = build_obs(ps, action, target_position, cfg.obs_noise, noise_key, self.ids, payload=cfg.payload)
       
        # Exponential decay with reset on disturbance
        prev_dynamic = state.metrics['dynamic']
        decay = jp.exp(-self.time_per_action / self.dynamic_decay_tau)
        dynamic_metric = jp.where(disturbance_flag > 0.0, 1.1, prev_dynamic * decay)
        

        reward = calc_reward(
            obs, ps.time, collision, out_of_bounds, action, angles, target_position,
            ps, max_thrust, cfg, dynamic_metric
        )

        # dont terminate ground collision on ground start
        if cfg.payload:
            ground_collision = jp.logical_and(
                ground_collision,
                jp.logical_or(
                    ps.time > 3, # allow 2 seconds for takeoff
                    ps.cvel[self.ids["payload_body_id"]][2] < -3.0,
                )
            )
        else:
            ground_collision = jp.logical_and(
                ground_collision,
                ps.time > 1, # allow 1 seconds for takeoff
            )

        collision = jp.logical_or(quad_collision, ground_collision)
        
        done = jp.logical_or(out_of_bounds, collision)
    
        
        done = done * 1.0

        # Apply random RPM jump disturbance
        p_jump = jp.clip(self.time_per_action / self.rpm_jump_interval_s, 0.0, 1.0)
        jump_key, delta_key = jax.random.split(noise_key)
        jump_flag = jax.random.bernoulli(jump_key, p=p_jump).astype(jp.float32)
        jump_delta = self.rpm_jump_std * jax.random.normal(delta_key, shape=filtered_rpm_proxy.shape)
        filtered_rpm_proxy = jp.clip(filtered_rpm_proxy + jump_flag * jump_delta, 0.0, jp.inf)

        metrics = {
            'time': ps.time,
            'reward': reward,
            'max_thrust': state.metrics['max_thrust'],
            'filtered_rpm_proxy': filtered_rpm_proxy,
            'motor_alpha': state.metrics['motor_alpha'],
            'dynamic': dynamic_metric,
        }
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, metrics=metrics)

