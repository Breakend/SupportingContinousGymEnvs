import os
import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_extensions.continuous.mujoco.wall_envs import WallEnvFactory
from gym_extensions.continuous.mujoco.gravity_envs import GravityEnvFactory
from gym.envs.mujoco.humanoid import HumanoidEnv, mass_center
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym_extensions.continuous.mujoco.perturbed_bodypart_env import ModifiedSizeEnvFactory

HumanoidWallEnv = lambda *args, **kwargs: WallEnvFactory(ModifiedHumanoidEnv)(
    model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoid.xml", ori_ind=-1, *args, **kwargs)

HumanoidStandupAndRunWallEnv = lambda *args, **kwargs: WallEnvFactory(HumanoidStandupAndRunEnv)(
    model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoidstandup.xml", ori_ind=-1, *args, **kwargs)

HumanoidGravityEnv = lambda *args, **kwargs: GravityEnvFactory(ModifiedHumanoidEnv)(
    model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoid.xml", *args, **kwargs)

HumanoidModifiedBodyPartSizeEnv = lambda *args, **kwargs: ModifiedSizeEnvFactory(ModifiedHumanoidEnv)(
    model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoid.xml", *args, **kwargs)


class ModifiedHumanoidEnv(HumanoidEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """

    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 5)
        utils.EzPickle.__init__(self)


class HumanoidWithSensorEnv(HumanoidEnv, utils.EzPickle):
    """
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we get sensor readouts with distances to the wall
    """

    def __init__(self, n_bins=10, **kwargs):
        self.n_bins = n_bins
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        obs = np.concatenate([
            HumanoidEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs


class HumanoidStandupWithSensorEnv(HumanoidStandupEnv, utils.EzPickle):
    """
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we get sensor readouts with distances to the wall
    """

    def __init__(self, n_bins=10, **kwargs):
        self.n_bins = n_bins
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        obs = np.concatenate([
            HumanoidStandupEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs


class HumanoidStandupAndRunEnv(HumanoidEnv, utils.EzPickle):
    """
    Environment that rewards standing up and the running, this is the combination of Humanoid-v1 and HumanoidStandup-v1
    """

    def __init__(self, model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoidstandup.xml", **kwargs):
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)

        pos_after_standup = self.sim.data.qpos[2]

        down = bool((pos_after_standup < 1.0) or (pos_after_standup > 2.0))

        alive_bonus = 5.0 if not down else 1.0

        data = self.sim.data

        uph_cost = (pos_after_standup - 0) / self.model.opt.timestep
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)

        reward = lin_vel_cost + uph_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        done = bool(False)
        return self._get_obs(), reward, done, dict(reward_linup=uph_cost, reward_quadctrl=-quad_ctrl_cost,
                                                   reward_impact=-quad_impact_cost, reward_alive=alive_bonus)


class HumanoidStandupAndRunEnvWithSensor(HumanoidStandupAndRunEnv, utils.EzPickle):
    """
    Environment that rewards standing up and the running, this is the combination of Humanoid-v1 and HumanoidStandup-v1, also adds empty sensor
    readouts to the state space to make compatible with transfer learning for the WallEnv variant
    """

    def __init__(self, model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoidstandup.xml", n_bins=10,
                 **kwargs):
        self.n_bins = n_bins
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        obs = np.concatenate([
            HumanoidStandupAndRunEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs
