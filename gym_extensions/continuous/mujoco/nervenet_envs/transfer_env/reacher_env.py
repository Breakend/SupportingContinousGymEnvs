#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       The reacher task
#   @author:
#       Tingwu (Wilson) Wang, July 22nd, 2017
# -----------------------------------------------------------------------------

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gym_extensions.continuous.mujoco.nervenet_envs.init_path as init_path
import os
import num2words


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    '''
        @brief:
            In the modifiedReacherEnv, we take out the cos and sin functions
            applied on the nervenet_envs.
            In this case, the env is compatible with all the other nervenet_envs
        @reward:
            normalize the distance reward to -1
            normalize the ctrl reward to -0.1
    '''

    def __init__(self, pod_number=2):

        # get the path of the nervenet_envs
        xml_name = 'Reacher' + self.get_env_num_str(pod_number) + '.xml'
        xml_path = os.path.join(os.path.join(init_path.get_base_dir(),
                                'nervenet_envs', 'assets', xml_name))
        xml_path = str(os.path.abspath(xml_path))

        # the environment coeff
        self.num_body = pod_number + 1
        self._task_indicator = -1.0

        self._ctrl_coeff = 2.0 / (self.num_body / 2 + 1)
        # norm the max penalty to be 1, max norm is self.num_body * 0.1 * 2
        self._dist_coeff = 2.0 / self.num_body

        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)
        utils.EzPickle.__init__(self)

    def step(self, a):
        vec = self.sim.data.site_xpos[0] - self.get_body_com("target")
        '''
        max is: 0.4
        reward_ctrl = - np.square(a).sum()
        reward_dist = self._task_indicator * \
            sum(np.square(vec)) * self._dist_coeff
        '''
        reward_ctrl = - np.square(a).sum() * self._ctrl_coeff
        reward_dist = -1 * np.linalg.norm(vec) * self._dist_coeff

        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, \
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.8

    def reset_model(self):
        # init qpos
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) \
            + self.init_qpos

        # get goal position
        self.goal = self.np_random.uniform(
            low=-.1 * self.num_body, high=.1 * self.num_body, size=2
        )
        '''
        length = self.np_random.uniform(
            low=0.0, high=0.1 * self.num_body, size=1
        )[0]
        theta = self.np_random.uniform(
            low=-3.1415, high=3.1415, size=1
        )[0]
        self.goal = [length * np.sin(theta), length * np.cos(theta)]
        '''
        qpos[-2:] = self.goal

        # init qvel
        qvel = self.init_qvel + \
            self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        if self._task_indicator < 0:
            qvel[-2:] = 0
        else:
            qvel[-2:] = self.np_random.uniform(low=-3, high=3, size=2)

        # set state
        self.set_state(qpos, qvel)
        self.set_color()

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:-2],
            self.sim.data.qvel.flat[:-2],
            (self.sim.data.site_xpos[0] - self.get_body_com("target"))[:2],
            [self._task_indicator],  # indicating task, -1: reacher, 1: avoider
            self.sim.data.qpos.flat[-2:]  # target pos
            # self.sim.data.qvel.flat[-2:]  # target vel
        ])

    def get_env_num_str(self, number):
        num_str = num2words.num2words(number)
        return num_str[0].upper() + num_str[1:]

    def set_color(self):
        reacher_id = self.model.geom_names.index('reacherIndicator')
        avoider_id = self.model.geom_names.index('avoiderIndicator')

        if self._task_indicator < 0:
            self.model.geom_size[reacher_id, 0] = 0.015
            self.model.geom_size[avoider_id, 0] = 0.001
        else:
            self.model.geom_size[reacher_id, 0] = 0.001
            self.model.geom_size[avoider_id, 0] = 0.015


class AvoiderEnv(ReacherEnv):
    '''
        @brief:
            In the modifiedReacherEnv, we take out the cos and sin functions
            applied on the nervenet_envs.
            In this case, the env is compatible with all the other nervenet_envs
    '''

    def __init__(self, pod_number=2):
        ReacherEnv.__init__(self, pod_number=pod_number)

    def reset_model(self):
        self._task_indicator = 1.0
        ReacherEnv.reset_model(self)  # reset the model
        return self._get_obs()


class SwitcherEnv(ReacherEnv):
    '''
        @brief:
            In the modifiedReacherEnv, we take out the cos and sin functions
            applied on the nervenet_envs.
            In this case, the env is compatible with all the other nervenet_envs
    '''

    def reset_model(self):
        self._task_indicator = float(self.np_random.randint(2) * 2 - 1)
        ReacherEnv.reset_model(self)  # reset the model
        return self._get_obs()


'''
    Reachers
'''


class ReacherZeroEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=0)


class ReacherOneEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=1)


class ReacherTwoEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=2)


class ReacherThreeEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=3)


class ReacherFourEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=4)


class ReacherFiveEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=5)


class ReacherSixEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=6)


class ReacherSevenEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=7)


class ReacherEightEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=8)


class ReacherNineEnv(ReacherEnv):

    def __init__(self):
        ReacherEnv.__init__(self, pod_number=9)


'''
    Avoiders
'''


class AvoiderZeroEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=0)


class AvoiderOneEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=1)


class AvoiderTwoEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=2)


class AvoiderThreeEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=3)


class AvoiderFourEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=4)


class AvoiderFiveEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=5)


class AvoiderSixEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=6)


class AvoiderSevenEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=7)


class AvoiderEightEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=8)


class AvoiderNineEnv(AvoiderEnv):

    def __init__(self):
        AvoiderEnv.__init__(self, pod_number=9)


'''
    Switchers
'''


class SwitcherZeroEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=0)


class SwitcherOneEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=1)


class SwitcherTwoEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=2)


class SwitcherThreeEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=3)


class SwitcherFourEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=4)


class SwitcherFiveEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=5)


class SwitcherSixEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=6)


class SwitcherSevenEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=7)


class SwitcherEightEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=8)


class SwitcherNineEnv(SwitcherEnv):

    def __init__(self):
        SwitcherEnv.__init__(self, pod_number=9)
