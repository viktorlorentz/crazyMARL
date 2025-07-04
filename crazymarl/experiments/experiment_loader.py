import asdf
import numpy as np
from crazymarl.observations.multi_quad_observation import get_obs_index_lookup

class Experiment:
    """
    Handles loading of ASDF file and provides lazy-loaded properties for observations,
    dones, dt, trajectory, and derived metrics while keeping the file open.
    Uses index lookup for flexible feature slicing across any number of quads.
    All convenience outputs are stacked over the quad dimension: (timesteps, runs, num_quads, feature_dim).
    """
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._af = asdf.open(file_path)
        self._obs = None
        self._dones = None
        self._dt = None
        self._trajectory = None
        self._first_dones = None
        self._payload_pos = None

        # load env config and build index lookup
        self._env_config = self._af['flights'][0]['metadata']['env_config']
        self.num_quads = self._env_config['num_quads']
        # global_ix not used directly here, but available
        self.global_ix, self.agent_ix = get_obs_index_lookup(self.num_quads)

        self.num_runs = self._af['flights'][0]['metadata']['num_envs']
        self.timesteps = self._af['flights'][0]['metadata']['timesteps']

        print(f"Experiment loaded from {file_path} with {self.num_quads} quads")


    def __del__(self):
        try:
            self._af.close()
        except Exception:
            pass

    # @property
    # def obs(self) -> np.ndarray:
    #     if self._obs is None:
    #         # load per-agent observations and concatenate to full flat obs
    #         flights = self._af['flights'][0]['agents']
    #         obs_list = []
    #         for i in range(self.num_quads):
    #             agent_key = f'agent_{i}'
    #             obs_i = np.array(flights[agent_key]['observations'])  # (T, runs, per_agent_dim)
    #             obs_list.append(obs_i)
    #         # concatenate along feature axis
    #         self._obs = np.concatenate(obs_list, axis=2)  # (T, runs, full_dim)
    #     return self._obs

    @property
    def dones(self) -> np.ndarray:
        if self._dones is None:
            self._dones = np.array(self._af['flights'][0]['global']['dones'])
        return self._dones

    @property
    def env_config(self) -> dict:
        return self._env_config

    @property
    def dt(self) -> float:
        if self._dt is None:
            freq = self.env_config['policy_freq']
            self._dt = 1.0 / freq
        return self._dt

    @property
    def trajectory(self) -> np.ndarray:
        if self._trajectory is None:
            self._trajectory = np.array(
                self._af['flights'][0]['global']['trajectory']
            )
        return self._trajectory

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.timesteps) * self.dt

    @property
    def first_dones(self) -> np.ndarray:
        if self._first_dones is None:
            self._first_dones = np.argmax(self.dones, axis=0)
        return self._first_dones

    @property
    def full_runs(self) -> np.ndarray:
        # number of runs where first done is last timestep
        return np.where(self.first_dones > self.first_dones.shape[0] - 1)[0]

    @property
    def failed_runs(self) -> np.ndarray:
        return np.where(self.first_dones <= self.first_dones.shape[0] - 1)[0]

    @property
    def agents(self) -> list:
        """List of agent identifiers"""
        return [f'agent_{i}' for i in range(self.num_quads)]

    def get_feature(self, feature: str, agent: int) -> np.ndarray:
        """
        Returns a specific feature slice for given agent: shape (timesteps, runs, dim).
        """
        name = f'agent_{agent}'
        idx = self.agent_ix[name][feature]
        agent_obs = np.array(self._af['flights'][0]['agents'][name]['observations'])   
        return agent_obs[:, :, idx]  # (timesteps, runs, feature_dim)

    def get_agent_obs(self, agent: int) -> np.ndarray:
        """
        Full mapped observation for one agent: shape (timesteps, runs, obs_dim_agent).
        """
        return self.get_feature('full_mapped_obs', agent)

    # --- stacked convenience properties ---

    @property
    def payload_error(self) -> np.ndarray:
        """
        Payload error for each quad: shape (timesteps, runs, num_quads, 3).
        """
        # get for agent 0 then tile across quads
        pe = self.get_feature('payload_error', 0)  # (T, R, 3)
        return np.tile(pe[:, :, None, :], (1, 1, self.num_quads, 1))

    @property
    def payload_linvel(self) -> np.ndarray:
        """
        Payload linear velocity for each quad: shape (timesteps, runs, num_quads, 3).
        """
        pl = self.get_feature('payload_linvel', 0)
        return np.tile(pl[:, :, None, :], (1, 1, self.num_quads, 1))

    @property
    def agent_rel_pos(self) -> np.ndarray:
        """
        Own relative position for all quads: shape (timesteps, runs, num_quads, 3).
        """
        arrs = [self.get_feature('agent_rel_pos', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def agent_rot_flat(self) -> np.ndarray:
        """
        Own rotation flattened for all quads: shape (timesteps, runs, num_quads, 9).
        """
        arrs = [self.get_feature('agent_rot_flat', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def agent_linvel(self) -> np.ndarray:
        """
        Own linear velocity for all quads: shape (timesteps, runs, num_quads, 3).
        """
        arrs = [self.get_feature('agent_linvel', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def agent_angvel(self) -> np.ndarray:
        """
        Own angular velocity for all quads: shape (timesteps, runs, num_quads, 3).
        """
        arrs = [self.get_feature('agent_angvel', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def agent_action(self) -> np.ndarray:
        """
        Own last action for all quads: shape (timesteps, runs, num_quads, 4).
        """
        arrs = [self.get_feature('agent_action', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def other_rel_pos(self) -> np.ndarray:
        """
        Relative positions of other quads for each agent: 
        shape (timesteps, runs, num_quads, 3*(num_quads-1)).
        """
        arrs = [self.get_feature('others_rel_pos', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def payload_pos(self) -> np.ndarray:
        """
        Absolute payload position in world frame per run: shape (timesteps, runs, 3).
        """
        if self._payload_pos is None:
            self._payload_pos = np.array(
                self._af['flights'][0]['global']['state']['payload_pos']
            )
        return self._payload_pos

    @property
    def quad_pos(self) -> np.ndarray:
        """
        Absolute quad positions in world frame: shape (timesteps, runs, num_quads, 3).
        """
        payload_world = self.payload_pos  # (T, runs, 3)
        rel = self.agent_rel_pos           # (T, runs, Q, 3)
        return payload_world[:, :, None, :] + rel  # (T, runs, Q, 3)
 

    def info(self):
        """
        Print detailed information about the experiment, including data shapes,
        run counts, and timing.
        """
        print(f"File: {self._file_path}")
        print(f"Quads: {self.num_quads}")
        print(f"Timesteps: {self.timesteps}")
        print(f"Runs: {self.num_runs}")
        print(f"DT: {self.dt}")
        print(f"Duration: {self.time[-1]:.3f}s")
        print(f"First done per run: {self.first_dones}")
        print(f"Full runs (>2000): {len(self.full_runs)}/{self.num_runs}")
        print(f" trajectory: {self.trajectory.shape if self.trajectory is not None else 'None'}")
        print("Sample feature shapes:")
        print(f" payload_error: {self.payload_error.shape}")
        print(f" payload_linvel: {self.payload_linvel.shape}")
        print(f" agent_rel_pos: {self.agent_rel_pos.shape}")
        print(f" agent_action: {self.agent_action.shape}")
        print(f" other_rel_pos: {self.other_rel_pos.shape}")
        print(f" quad_pos: {self.quad_pos.shape}")


