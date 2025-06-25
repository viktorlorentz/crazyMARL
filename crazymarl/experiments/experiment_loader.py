import asdf
import numpy as np

class Experiment:
    """
    Handles loading of ASDF file and provides lazy-loaded properties for observations,
    dones, dt, trajectory, and derived metrics while keeping the file open.
    """
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._af = asdf.open(file_path)
        self._obs = None
        self._dones = None
        self._dt = None
        self._trajectory = None
        self._first_dones = None
        self._env_config = None
        print(f"Experiment loaded from {file_path}")

    def __del__(self):
        try:
            self._af.close()
        except Exception:
            pass

    @property
    def obs(self) -> np.ndarray:
        if self._obs is None:
            self._obs = np.array(self._af['flights'][0]['agents']['agent_0']['observations'])
        return self._obs

    @property
    def dones(self) -> np.ndarray:
        if self._dones is None:
            self._dones = np.array(self._af['flights'][0]['global']['dones'])
        return self._dones
    
    @property
    def env_config(self) -> dict:
        if self._env_config is None:
            self._env_config = self._af['flights'][0]['metadata']['env_config']
        return self._env_config

    @property
    def dt(self) -> float:
        if self._dt is None:
            self.policy_freq = self.env_config['policy_freq']
            self._dt =  1.0 / self.policy_freq
        return self._dt

    @property
    def trajectory(self) -> np.ndarray:
        if self._trajectory is None:
            self._trajectory = np.array(self._af['flights'][0]['global']['trajectory'])
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
    def payload(self) -> np.ndarray:
        return self.obs[:, :, 0:3]

    @property
    def payload_velocity(self) -> np.ndarray:
        return self.obs[:, :, 3:6]

    @property
    def other_quads(self) -> np.ndarray:
        return self.obs[:, :, 6:9]

    @property
    def q_pos(self) -> np.ndarray:
        return self.obs[:, :, 9:12]

    @property
    def q_rot_mat(self) -> np.ndarray:
        return self.obs[:, :, 12:21]

    @property
    def q_linvel(self) -> np.ndarray:
        return self.obs[:, :, 21:24]

    @property
    def q_angvel(self) -> np.ndarray:
        return self.obs[:, :, 24:27]

    @property
    def q_linacc(self) -> np.ndarray:
        return self.obs[:, :, 27:30]

    @property
    def q_angacc(self) -> np.ndarray:
        return self.obs[:, :, 30:33]

    @property
    def q_last_action(self) -> np.ndarray:
        return self.obs[:, :, 33:36]

    def info(self):
        """
        Print detailed information about the experiment, including data shapes,
        run counts, and timing.
        """
        print(f"File path: {self._file_path}")
        print(f"Timesteps: {self.obs.shape[0]}")
        print(f"Runs: {self.obs.shape[1]}")
        print(f"Time step (dt): {self.dt}")
        print(f"Total duration: {self.time[-1]:.3f} seconds")
        print(f"First done indices per run: {self.first_dones}")
        n_full = len(self.full_runs)
        print(f"Full runs (>2000 steps): {n_full}/{self.obs.shape[1]} ({n_full/self.obs.shape[1]*100:.2f}%)")
        print("Observation channel shapes:")
        print(f"  payload: {self.payload.shape}")
        print(f"  payload_velocity: {self.payload_velocity.shape}")
        print(f"  other_quads: {self.other_quads.shape}")
        print(f"  q_pos: {self.q_pos.shape}")
        print(f"  q_rot_mat: {self.q_rot_mat.shape}")
        print(f"  q_linvel: {self.q_linvel.shape}")
        print(f"  q_angvel: {self.q_angvel.shape}")
        print(f"  q_last_action: {self.q_last_action.shape}")
