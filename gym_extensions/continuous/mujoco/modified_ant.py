from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_extensions.continuous.mujoco.wall_envs import MazeFactory
from gym_extensions.continuous.mujoco.gravity_envs import GravityEnvFactory
from gym_extensions.continuous.mujoco.food_envs import SingleFoodEnvFactory
from gym.envs.mujoco.ant import AntEnv

import os
import gym

AntGravityEnv = lambda *args, **kwargs: GravityEnvFactory(ModifiedAntEnv)(
    model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/ant.xml", *args, **kwargs)

AntMaze = lambda *args, **kwargs: MazeFactory(ModifiedAntEnv)(
    model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/ant.xml", ori_ind=0, *args, **kwargs)

AntSingleFoodEnv = lambda *args, **kwargs: SingleFoodEnvFactory(ModifiedAntEnv)(
    model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/ant.xml", *args, **kwargs)

class ModifiedAntEnv(AntEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """

    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4)
        utils.EzPickle.__init__(self)
