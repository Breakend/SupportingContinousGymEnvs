import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
import math

import numpy as np
import gym
import random
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import six


def ModifiedMassEnvFactory(class_type):
    """class_type should be an OpenAI gym type"""

    class ModifiedMassEnv(class_type, utils.EzPickle):
        """
        Allows the gravity to be changed by the
        """

        def __init__(
                self,
                model_path,
                body_part="torso",
                mass_scale=1.0,
                *args,
                **kwargs):
            class_type.__init__(self, model_path=model_path)
            utils.EzPickle.__init__(self)

            # make sure we're using a proper OpenAI gym Mujoco Env
            assert isinstance(self, mujoco_env.MujocoEnv)

            self.model.body_mass = self.get_and_modify_bodymass(body_part, mass_scale)
            self.model._compute_subtree()
            self.model.forward()

        def get_and_modify_bodymass(self, body_name, scale):
            idx = self.model.body_names.index(six.b(body_name))
            temp = np.copy(self.model.body_mass)
            temp[idx] *= scale
            return temp

    return ModifiedMassEnv


def ModifiedSizeEnvFactory(class_type):
    """class_type should be an OpenAI gym type"""

    class ModifiedSizeEnv(class_type, utils.EzPickle):
        """
        Allows the gravity to be changed by the
        """

        def __init__(
                self,
                model_path,
                body_parts=["torso"],
                size_scale=1.0,
                *args,
                **kwargs):
            assert isinstance(self, mujoco_env.MujocoEnv)

            # find the body_part we want
            tree = ET.parse(model_path)
            for body_part in body_parts:
                # torso = tree.find(".//body[@name='%s']" % body_part)

                # grab the geoms
                geom = tree.find(".//geom[@name='%s']" % body_part)

                sizes = [float(x) for x in geom.attrib["size"].split(" ")]

                # rescale
                geom.attrib["size"] = " ".join(
                    [str(x * size_scale) for x in sizes])  # the first one should always be the thing we want.

                # TODO: in the future we want to also be able to make it longer or shorter, but this requires propagation of the fromto attribute
                # so like a middle part isn't super long but the other parts connect at the same spot.

                # fromto = []
                # for x in geoms[0].attrib["fromto"].split(" "):
                #     fromto.append(float(x))
                # fromto = [x*length_scale for x in fromto]
                # geoms[0].attrib["fromto"] = str() * length_scale) # the first one should always be the thing we want.

            # create new xml
            _, file_path = tempfile.mkstemp(suffix=".xml", text=True)
            tree.write(file_path)

            # load the modified xml
            class_type.__init__(self, model_path=file_path)
            utils.EzPickle.__init__(self)

        # def get_and_modify_bodysize(self, body_name, scale):
        #     idx = self.model.body_names.index(six.b(body_name))
        #     temp = np.copy(self.model.geom_size)
        #     temp[idx] *= scale
        #     return temp

    return ModifiedSizeEnv
