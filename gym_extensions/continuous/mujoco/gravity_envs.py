from gym import utils
from gym.envs.mujoco import mujoco_env

""" As of 18/07/2019
Check this out!
This can be good reference to modify some attributes in `mujoco_py`
https://gist.github.com/machinaut/9bd1473c763554086c176d39062700b0
"""


def GravityEnvFactory(class_type):
    """class_type should be an OpenAI gym type"""

    class GravityEnv(class_type, utils.EzPickle):
        """
        Allows the gravity to be changed by the
        """

        def __init__(
                self,
                model_path,
                gravity=-9.81,
                *args,
                **kwargs):
            class_type.__init__(self, model_path=model_path)
            utils.EzPickle.__init__(self)

            # make sure we're using a proper OpenAI gym Mujoco Env
            assert isinstance(self, mujoco_env.MujocoEnv)

            self.model.opt.gravity[0] = 0.
            self.model.opt.gravity[1] = 0.
            self.model.opt.gravity[2] = gravity * 3

    return GravityEnv
