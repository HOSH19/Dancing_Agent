import numpy as np
import mujoco
import gymnasium as gym

FOOT_NAMES   = ("right_foot", "left_foot")
FLOOR        = 0.055   # above resting height (~0.05m)
SWING_CAP    = 0.12    # max height rewarded (realistic stride)
CONTACT_N    = 1.0     # N threshold for stance detection
SWING_W      = 20.0
UPRIGHT_W    = 2.0


class WalkEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        m = env.unwrapped.model
        self._foot_ids = [m.body(name).id for name in FOOT_NAMES]
        self._torso_id = m.body("torso").id

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        swing   = self._swing()
        upright = self._upright()
        info["reward_clearance"] = swing
        info["reward_upright"]   = upright
        return obs, reward + SWING_W * swing + UPRIGHT_W * upright, terminated, truncated, info

    def _contact_forces(self):
        d, m = self.env.unwrapped.data, self.env.unwrapped.model
        forces = {fid: 0.0 for fid in self._foot_ids}
        buf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            mujoco.mj_contactForce(m, d, i, buf)
            for fid in self._foot_ids:
                if b1 == fid or b2 == fid:
                    forces[fid] += abs(buf[0])
        return forces

    def _swing(self):
        forces  = self._contact_forces()
        grounded = [fid for fid, f in forces.items() if f > CONTACT_N]
        if len(grounded) != 1:
            return 0.0
        swing_fid = next(fid for fid in self._foot_ids if fid not in grounded)
        h = self.env.unwrapped.data.xpos[swing_fid, 2]
        return min(max(0.0, h - FLOOR), SWING_CAP - FLOOR)

    def _upright(self):
        return self.env.unwrapped.data.xmat[self._torso_id, 8]
