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

        # load env config and build index lookup
        self._env_config = self._af['flights'][0]['metadata']['env_config']
        self.num_quads = self._env_config['num_quads']
        # global_ix not used directly here, but available
        _, self.agent_ix = get_obs_index_lookup(self.num_quads)

        print(f"Experiment loaded from {file_path} with {self.num_quads} quads")

    def __del__(self):
        try:
            self._af.close()
        except Exception:
            pass

    @property
    def obs(self) -> np.ndarray:
        if self._obs is None:
            # assume 'agent_0' stores full flattened obs
            self._obs = np.array(
                self._af['flights'][0]['agents']['agent_0']['observations']
            )  # shape (timesteps, runs, features)
        return self._obs

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
        return np.arange(self.obs.shape[0]) * self.dt

    @property
    def first_dones(self) -> np.ndarray:
        if self._first_dones is None:
            self._first_dones = np.argmax(self.dones, axis=0)
        return self._first_dones

    @property
    def full_runs(self) -> np.ndarray:
        return np.where(self.first_dones > 2000)[0]

    @property
    def failed_runs(self) -> np.ndarray:
        return np.where(self.first_dones <= 2000)[0]

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
        return self.obs[:, :, idx]

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
    def own_rel_pos(self) -> np.ndarray:
        """
        Own relative position for all quads: shape (timesteps, runs, num_quads, 3).
        """
        arrs = [self.get_feature('own_rel_pos', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def own_rot_flat(self) -> np.ndarray:
        """
        Own rotation flattened for all quads: shape (timesteps, runs, num_quads, 9).
        """
        arrs = [self.get_feature('own_rot_flat', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def own_linvel(self) -> np.ndarray:
        """
        Own linear velocity for all quads: shape (timesteps, runs, num_quads, 3).
        """
        arrs = [self.get_feature('own_linvel', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def own_angvel(self) -> np.ndarray:
        """
        Own angular velocity for all quads: shape (timesteps, runs, num_quads, 3).
        """
        arrs = [self.get_feature('own_angvel', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    @property
    def own_action(self) -> np.ndarray:
        """
        Own last action for all quads: shape (timesteps, runs, num_quads, 4).
        """
        arrs = [self.get_feature('own_action', i) for i in range(self.num_quads)]
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
        Absolute payload position relative to target: shape (timesteps, runs, 3).
        """
        # target position from environment config
        tp = np.array(self.env_config['target_position'])  # (3,)
        # trajectory is payload world positions (timesteps, runs, 3)
        return self.trajectory - tp[None, None, :]

    @property
    def quad_pos(self) -> np.ndarray:
        """
        Absolute quad positions relative to target: shape (timesteps, runs, num_quads, 3).
        """
        # target position
        tp = np.array(self.env_config['target_position'])  # (3,)
        # payload world positions
        payload_world = self.trajectory  # (T, runs, 3)
        # quad positions relative to payload
        rel = self.own_rel_pos        # (T, runs, Q, 3)
        # quad world positions
        quad_world = payload_world[:, :, None, :] + rel  # (T, runs, Q, 3)
        return quad_world - tp[None, None, None, :]

        """
        Relative positions of other quads for each agent: 
        shape (timesteps, runs, num_quads, 3*(num_quads-1)).
        """
        arrs = [self.get_feature('others_rel_pos', i) for i in range(self.num_quads)]
        return np.stack(arrs, axis=2)

    def info(self):
        """
        Print detailed information about the experiment, including data shapes,
        run counts, and timing.
        """
        print(f"File: {self._file_path}")
        print(f"Quads: {self.num_quads}")
        print(f"Timesteps: {self.obs.shape[0]}")
        print(f"Runs: {self.obs.shape[1]}")
        print(f"DT: {self.dt}")
        print(f"Duration: {self.time[-1]:.3f}s")
        print(f"First done per run: {self.first_dones}")
        print(f"Full runs (>2000): {len(self.full_runs)}/{self.obs.shape[1]}")
        print("Sample feature shapes:")
        print(f" payload_error: {self.payload_error.shape}")
        print(f" payload_linvel: {self.payload_linvel.shape}")
        print(f" own_rel_pos: {self.own_rel_pos.shape}")
        print(f" own_action: {self.own_action.shape}")
