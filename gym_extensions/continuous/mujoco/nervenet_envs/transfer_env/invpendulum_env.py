#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       The snake nervenet_envs.
#   @author:
#       Tingwu (Wilson) Wang, Aug. 30nd, 2017
# -----------------------------------------------------------------------------

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gym_extensions.continuous.mujoco.nervenet_envs.init_path as init_path
import os
import num2words


class InvPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    '''
        @brief:
            In the Swimmer, we define a task that is designed to test the
            power of transfer learning for gated graph neural network.
    '''

    def __init__(self, pod_number=3):

        # get the path of the nervenet_envs
        xml_name = 'InvPendulum' + self.get_env_num_str(pod_number) + '.xml'
        xml_path = os.path.join(os.path.join(init_path.get_base_dir(),
                                'nervenet_envs', 'assets', xml_name))
        xml_path = str(os.path.abspath(xml_path))
        self.num_body = pod_number

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def get_env_num_str(self, number):
        num_str = num2words.num2words(number)
        return num_str[0].upper() + num_str[1:]

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        # distance penalty
        x, _, y = self.model.data.site_xpos[0]  # the marker
        dist_penalty = 0.01 * x ** 2 + (y - 0.6 * self.num_body - 0.8) ** 2

        # velocity penalty
        v1 = self.model.data.qvel[1]  # the root joint
        v2 = self.model.data.qvel[2:]  # the other joints
        vel_penalty = 1e-3 * v1**2 + sum(5e-3 * v2 ** 2) * 2 / self.num_body

        # alive bonus
        alive_bonus = 10

        r = (alive_bonus - dist_penalty - vel_penalty)[0]
        done = bool(y <= 0.4 * self.num_body)  # 2 is 1, 1.2
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],  # cart x pos
            self.model.data.qpos[1:],
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(
                low=-.1, high=.1, size=self.model.nq
            ),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent * 0.8
        v.cam.lookat[2] += 3


'''
    the following nervenet_envs are just models with even number legs
'''


class InvPendulumOneEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=1)


class InvPendulumTwoEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=2)


class InvPendulumThreeEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=3)


class InvPendulumFourEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=4)


class InvPendulumFiveEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=5)


class InvPendulumSixEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=6)


class InvPendulumSevenEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=7)


class InvPendulumEightEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=8)


class InvPendulumNineEnv(InvPendulumEnv):

    def __init__(self):
        InvPendulumEnv.__init__(self, pod_number=9)
