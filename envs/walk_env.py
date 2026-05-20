import gymnasium as gym

FOOT_NAMES = ("right_foot", "left_foot")
CLEARANCE_THRESHOLD = 0.02  # m — lowered so signal fires earlier in swing
CLEARANCE_W = 40.0           # increased so lift bonus is meaningful vs healthy=5.0
UPRIGHT_W = 3.0              # torso z-axis alignment with world up; 3.0 * 1.0 = 3.0/step max


class WalkEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        m = env.unwrapped.model
        self._foot_ids = [m.body(name).id for name in FOOT_NAMES]
        self._torso_id = m.body("torso").id

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        clearance = self._clearance()
        upright = self._upright()
        info["reward_clearance"] = clearance
        info["reward_upright"] = upright
        return obs, reward + CLEARANCE_W * clearance + UPRIGHT_W * upright, terminated, truncated, info

    def _clearance(self):
        d = self.env.unwrapped.data
        return sum(max(0.0, d.xpos[fid, 2] - CLEARANCE_THRESHOLD) for fid in self._foot_ids)

    def _upright(self):
        # xmat[torso_id, 8] = R[2,2] = dot(body_z, world_z), 1.0 when perfectly upright
        return self.env.unwrapped.data.xmat[self._torso_id, 8]
