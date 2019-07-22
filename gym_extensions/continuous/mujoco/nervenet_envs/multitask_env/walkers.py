#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       Several Walkers
#   @author:
#       Tingwu (Wilson) Wang, Nov. 22nd, 2017
# -----------------------------------------------------------------------------

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gym_extensions.continuous.mujoco.nervenet_envs.init_path as init_path
import os
import num2words


def modify_xml(xml_name, num):
    if num is not None:
        if num <= 5:
            xml_name = xml_name.replace('.xml', num2words.num2words(num) + '.xml')
            # xml_name = 'mass/' + xml_name
            xml_name = 'strength/' + xml_name
        elif num <= 10:
            num -= 5
            xml_name = xml_name.replace('.xml', num2words.num2words(num) + '.xml')
            xml_name = 'strength/' + xml_name
        elif num <= 15:
            num -= 10
            xml_name = xml_name.replace('.xml', num2words.num2words(num) + '.xml')
            xml_name = 'length/' + xml_name
        else:
            raise NotImplementedError
        # print xml_name
    return xml_name


class WalkersHalfhumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, num=None):

        # get the path of the nervenet_envs
        xml_name = 'WalkersHalfhumanoid.xml'
        xml_name = modify_xml(xml_name, num)
        xml_path = os.path.join(os.path.join(init_path.get_base_dir(),
                                'nervenet_envs', 'assets', xml_name))
        xml_path = str(os.path.abspath(xml_path))
        self.num = num

        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20


class WalkersOstrichEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, num=None):

        # get the path of the nervenet_envs
        xml_name = 'WalkersOstrich.xml'
        xml_name = modify_xml(xml_name, num)
        xml_path = os.path.join(os.path.join(init_path.get_base_dir(),
                                'nervenet_envs', 'assets', xml_name))
        xml_path = str(os.path.abspath(xml_path))
        self.num = num

        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0 and
                    self.sim.data.site_xpos[0, 2] > 1.1)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20


class WalkersHopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, num=None):

        xml_name = 'WalkersHopper.xml'
        xml_name = modify_xml(xml_name, num)
        xml_path = os.path.join(os.path.join(init_path.get_base_dir(),
                                'nervenet_envs', 'assets', xml_name))
        self.num = num

        xml_path = str(os.path.abspath(xml_path))
        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20


class WalkersHalfcheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, num=None):

        xml_name = 'WalkersHalfcheetah.xml'
        xml_name = modify_xml(xml_name, num)
        xml_path = os.path.join(os.path.join(init_path.get_base_dir(),
                                'nervenet_envs', 'assets', xml_name))
        xml_path = str(os.path.abspath(xml_path))
        self.num = num

        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        alive_bonus = 1.0
        reward += alive_bonus
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    self.sim.data.site_xpos[2, 2] > 1.2 and
                    self.sim.data.site_xpos[0, 2] > 0.7 and
                    self.sim.data.site_xpos[1, 2] > 0.7
                    )

        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class WalkersFullcheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, num=None):

        xml_name = 'WalkersFullcheetah.xml'
        xml_name = modify_xml(xml_name, num)
        xml_path = os.path.join(os.path.join(init_path.get_base_dir(),
                                'nervenet_envs', 'assets', xml_name))
        xml_path = str(os.path.abspath(xml_path))
        self.num = num

        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        alive_bonus = 1
        reward += alive_bonus
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    self.sim.data.site_xpos[0, 2] > 0.7 and
                    self.sim.data.site_xpos[1, 2] > 0.7
                    )
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class WalkersKangarooEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):

        # get the path of the nervenet_envs
        xml_name = 'WalkersKangaroo.xml'
        xml_name = modify_xml(xml_name)
        xml_path = os.path.join(os.path.join(init_path.get_base_dir(),
                                'nervenet_envs', 'assets', xml_name))
        xml_path = str(os.path.abspath(xml_path))

        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt) / 2.0
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0 and self.sim.data.site_xpos[0, 2] > 0.8
                    and self.sim.data.site_xpos[0, 2] < 1.6)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20


# robust env
class WalkersHalfhumanoidzeroEnv(WalkersHalfhumanoidEnv):

    def __init__(self):
        super(WalkersHalfhumanoidzeroEnv, self).__init__(num=0)


class WalkersOstrichzeroEnv(WalkersOstrichEnv):

    def __init__(self):
        super(WalkersOstrichzeroEnv, self).__init__(num=0)


class WalkersHopperzeroEnv(WalkersHopperEnv):

    def __init__(self):
        super(WalkersHopperzeroEnv, self).__init__(num=0)


class WalkersHalfcheetahzeroEnv(WalkersHalfcheetahEnv):

    def __init__(self):
        super(WalkersHalfcheetahzeroEnv, self).__init__(num=0)


class WalkersFullcheetahzeroEnv(WalkersFullcheetahEnv):

    def __init__(self):
        super(WalkersFullcheetahzeroEnv, self).__init__(num=0)


class WalkersHalfhumanoidfiveEnv(WalkersHalfhumanoidEnv):

    def __init__(self):
        super(WalkersHalfhumanoidfiveEnv, self).__init__(num=5)


class WalkersOstrichfiveEnv(WalkersOstrichEnv):

    def __init__(self):
        super(WalkersOstrichfiveEnv, self).__init__(num=5)


class WalkersHopperfiveEnv(WalkersHopperEnv):

    def __init__(self):
        super(WalkersHopperfiveEnv, self).__init__(num=5)


class WalkersHalfcheetahfiveEnv(WalkersHalfcheetahEnv):

    def __init__(self):
        super(WalkersHalfcheetahfiveEnv, self).__init__(num=5)


class WalkersFullcheetahfiveEnv(WalkersFullcheetahEnv):

    def __init__(self):
        super(WalkersFullcheetahfiveEnv, self).__init__(num=5)


class WalkersHalfhumanoidfourEnv(WalkersHalfhumanoidEnv):

    def __init__(self):
        super(WalkersHalfhumanoidfourEnv, self).__init__(num=4)


class WalkersOstrichfourEnv(WalkersOstrichEnv):

    def __init__(self):
        super(WalkersOstrichfourEnv, self).__init__(num=4)


class WalkersHopperfourEnv(WalkersHopperEnv):

    def __init__(self):
        super(WalkersHopperfourEnv, self).__init__(num=4)


class WalkersHalfcheetahfourEnv(WalkersHalfcheetahEnv):

    def __init__(self):
        super(WalkersHalfcheetahfourEnv, self).__init__(num=4)


class WalkersFullcheetahfourEnv(WalkersFullcheetahEnv):

    def __init__(self):
        super(WalkersFullcheetahfourEnv, self).__init__(num=4)


class WalkersHalfhumanoidthreeEnv(WalkersHalfhumanoidEnv):

    def __init__(self):
        super(WalkersHalfhumanoidthreeEnv, self).__init__(num=3)


class WalkersOstrichthreeEnv(WalkersOstrichEnv):

    def __init__(self):
        super(WalkersOstrichthreeEnv, self).__init__(num=3)


class WalkersHopperthreeEnv(WalkersHopperEnv):

    def __init__(self):
        super(WalkersHopperthreeEnv, self).__init__(num=3)


class WalkersHalfcheetahthreeEnv(WalkersHalfcheetahEnv):

    def __init__(self):
        super(WalkersHalfcheetahthreeEnv, self).__init__(num=3)


class WalkersFullcheetahthreeEnv(WalkersFullcheetahEnv):

    def __init__(self):
        super(WalkersFullcheetahthreeEnv, self).__init__(num=3)


class WalkersHalfhumanoidtwoEnv(WalkersHalfhumanoidEnv):

    def __init__(self):
        super(WalkersHalfhumanoidtwoEnv, self).__init__(num=2)


class WalkersOstrichtwoEnv(WalkersOstrichEnv):

    def __init__(self):
        super(WalkersOstrichtwoEnv, self).__init__(num=2)


class WalkersHoppertwoEnv(WalkersHopperEnv):

    def __init__(self):
        super(WalkersHoppertwoEnv, self).__init__(num=2)


class WalkersHalfcheetahtwoEnv(WalkersHalfcheetahEnv):

    def __init__(self):
        super(WalkersHalfcheetahtwoEnv, self).__init__(num=2)


class WalkersFullcheetahtwoEnv(WalkersFullcheetahEnv):

    def __init__(self):
        super(WalkersFullcheetahtwoEnv, self).__init__(num=2)


class WalkersHalfhumanoidoneEnv(WalkersHalfhumanoidEnv):

    def __init__(self):
        super(WalkersHalfhumanoidoneEnv, self).__init__(num=1)


class WalkersOstrichoneEnv(WalkersOstrichEnv):

    def __init__(self):
        super(WalkersOstrichoneEnv, self).__init__(num=1)


class WalkersHopperoneEnv(WalkersHopperEnv):

    def __init__(self):
        super(WalkersHopperoneEnv, self).__init__(num=1)


class WalkersHalfcheetahoneEnv(WalkersHalfcheetahEnv):

    def __init__(self):
        super(WalkersHalfcheetahoneEnv, self).__init__(num=1)


class WalkersFullcheetahoneEnv(WalkersFullcheetahEnv):

    def __init__(self):
        super(WalkersFullcheetahoneEnv, self).__init__(num=1)
