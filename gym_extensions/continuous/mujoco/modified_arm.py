import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os.path as osp
from gym_extensions.continuous.mujoco.gravity_envs import GravityEnvFactory
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.striker import StrikerEnv

import os
import gym
import random
import six


class PusherFullRange(PusherEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """

    def __init__(self, model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/pusher.xml", **kwargs):
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

        # make sure we're using a proper OpenAI gym Mujoco Env
        assert isinstance(self, mujoco_env.MujocoEnv)

        temp_mat = self.get_and_modify_joint_range('r_shoulder_pan_joint')

        for i in range(temp_mat.shape[0]):
            self.model.jnt_range[i][0] = temp_mat[i][0]
            self.model.jnt_range[i][1] = temp_mat[i][1]

    def get_and_modify_joint_range(self, body_name, new_array=np.array([-.854, 2.714602])):
        idx = self.model.joint_names.index(body_name)
        temp = np.copy(self.model.jnt_range)
        temp[idx] = new_array
        return temp

    def reset_model(self):
        qpos = self.init_qpos

        goal_pos_map = {'left': np.array([-0.05, -1.1230]), 'right': np.array([0.0, 0.0])}
        object_start_pos_map = {'left': np.array([[-0.3, 0.0], [-1.1, -.8]]),
                                'right': np.array([[-0.3, 0.0], [-0.2, 0.2]])}
        position = random.choice(['left', 'right'])

        self.goal_pos = goal_pos_map[position]
        object_start_range = object_start_pos_map[position]

        while True:

            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=object_start_range[0][0], high=object_start_range[0][1], size=1),
                self.np_random.uniform(low=object_start_range[1][0], high=object_start_range[1][1], size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()


class PusherLeftSide(PusherEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """

    def __init__(self, model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/pusher.xml", **kwargs):
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

        # make sure we're using a proper OpenAI gym Mujoco Env
        assert isinstance(self, mujoco_env.MujocoEnv)

        temp_mat = self.get_and_modify_joint_range('r_shoulder_pan_joint')

        for i in range(temp_mat.shape[0]):
            self.model.jnt_range[i][0] = temp_mat[i][0]
            self.model.jnt_range[i][1] = temp_mat[i][1]

    def get_and_modify_joint_range(self, body_name, new_array=np.array([-.854, 2.714602])):
        idx = self.model.joint_names.index(body_name)
        temp = np.copy(self.model.jnt_range)
        temp[idx] = new_array
        return temp

    def reset_model(self):
        qpos = self.init_qpos
        self.goal_pos = np.array([-0.05, -1.1230])

        while True:
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-1.1, high=-.8, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()


class PusherMovingGoalEnv(PusherEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """

    def __init__(self, model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/pusher.xml", **kwargs):
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.concatenate([
            self.np_random.uniform(low=-0.2, high=0, size=1),
            self.np_random.uniform(low=-0.1, high=0.3, size=1)])

        while True:
            self.cylinder_pos = np.concatenate([
                self.np_random.uniform(low=-0.3, high=0, size=1),
                self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()


class StrikerMovingStartStateEnv(StrikerEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """

    def __init__(self, model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/striker.xml", **kwargs):
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.1
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

    def reset_model(self):
        self._min_strike_dist = np.inf
        self._striked = False
        self._strike_pos = None

        qpos = self.init_qpos

        self.ball = np.concatenate([
            self.np_random.uniform(low=0.43, high=0.55, size=1),
            self.np_random.uniform(low=-0.05, high=-.3, size=1)])

        while True:
            self.goal = np.concatenate([
                self.np_random.uniform(low=0.15, high=0.7, size=1),
                self.np_random.uniform(low=0.1, high=1.0, size=1)])
            if np.linalg.norm(self.ball - self.goal) > 0.17:
                break

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1,
                                                       size=self.model.nv)
        qvel[7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()
