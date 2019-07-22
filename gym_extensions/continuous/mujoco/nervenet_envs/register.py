# --environments---------------------------------------------------------------------------
#   @brief:
#       register the environments here
#   @author:
#       Tingwu Wang, July 3rd, 2017
# -----------------------------------------------------------------------------

from gym.envs.registration import register
import num2words
import gym_extensions.continuous.mujoco.nervenet_envs.asset_generator as asset_generator
import numpy as np

MAX_EPISODE_STEPS_DICT = {
    'Centipede': 1000,
    'CpCentipede': 1000,
    'Snake': 1000,
    'Reacher': 50,
}
REWARD_THRESHOLD = {
    'Centipede': 6000.0,
    'CpCentipede': 6000.0,
    'Snake': 360.0,
    'Reacher': -3.75,
}

# walker list
MULTI_TASK_DICT = {
    'MultiWalkers-v1':
        ['WalkersHopper-v1', 'WalkersHalfhumanoid-v1', 'WalkersHalfcheetah-v1',
         'WalkersFullcheetah-v1', 'WalkersOstrich-v1'],
    # just for implementation, only one agent will be run
    'MultiWalkers2Kangaroo-v1':
        ['WalkersHopper-v1', 'WalkersHalfhumanoid-v1', 'WalkersHalfcheetah-v1',
         'WalkersFullcheetah-v1', 'WalkersKangaroo-v1'],
}

# test the robustness of agents
NUM_ROBUSTNESS_AGENTS = 5
ROBUSTNESS_TASK_DICT = {}
for i_agent in range(1, NUM_ROBUSTNESS_AGENTS + 1):
    ROBUSTNESS_TASK_DICT.update({
        'MultiWalkers' + num2words.num2words(i_agent) + '-v1':
            ['WalkersHopper' + num2words.num2words(i_agent) + '-v1',
             'WalkersHalfhumanoid' + num2words.num2words(i_agent) + '-v1',
             'WalkersHalfcheetah' + num2words.num2words(i_agent) + '-v1',
             'WalkersFullcheetah' + num2words.num2words(i_agent) + '-v1',
             'WalkersOstrich' + num2words.num2words(i_agent) + '-v1'],
    })
MULTI_TASK_DICT.update(ROBUSTNESS_TASK_DICT)


name_list = []  # record all the environments available

# register the transfer tasks
for env_title, env in ROBUSTNESS_TASK_DICT.items():

    for i_env in env:
        file_name = 'gym_extensions.continuous.mujoco.nervenet_envs.multitask_env.walkers:'

        # WalkersHopperone-v1, WalkersHopperoneEnv
        entry_point = file_name + i_env.replace('-v1', 'Env')

        register(
            id=i_env,
            entry_point=entry_point,
            max_episode_steps=1000,
            reward_threshold=6000,
        )
        # register the environment name in the name list
        name_list.append(i_env)

# register the robustness tasks
for env in asset_generator.TASK_DICT:
    file_name = 'gym_extensions.continuous.mujoco.nervenet_envs.transfer_env.' + env.lower() + '_env:'

    for i_part in np.sort(asset_generator.TASK_DICT[env]):
        # NOTE, the order in the name_list actually matters (needed during
        # transfer learning)
        registered_name = env + num2words.num2words(i_part)[0].upper() + num2words.num2words(i_part)[1:]
        registered_name = registered_name.replace(' ', '')
        entry_point = file_name + registered_name + 'Env'
        register(
            id=(registered_name + '-v1'),
            entry_point=entry_point,
            max_episode_steps=MAX_EPISODE_STEPS_DICT[env],
            reward_threshold=REWARD_THRESHOLD[env],
        )
        # register the environment name in the name list
        name_list.append(registered_name + '-v1')

# register AntS-v1
register(
    id='AntS-v1',
    entry_point='gym_extensions.continuous.mujoco.nervenet_envs.transfer_env.antS:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000,
)

# register the walkers for multi-task learning
register(
    id='WalkersHopper-v1',
    entry_point='gym_extensions.continuous.mujoco.nervenet_envs.multitask_env.walkers:WalkersHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id='WalkersHalfhumanoid-v1',
    entry_point='gym_extensions.continuous.mujoco.nervenet_envs.multitask_env.walkers:WalkersHalfhumanoidEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id='WalkersHalfcheetah-v1',
    entry_point='gym_extensions.continuous.mujoco.nervenet_envs.multitask_env.walkers:WalkersHalfcheetahEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id='WalkersFullcheetah-v1',
    entry_point='gym_extensions.continuous.mujoco.nervenet_envs.multitask_env.walkers:WalkersFullcheetahEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id='WalkersOstrich-v1',
    entry_point='gym_extensions.continuous.mujoco.nervenet_envs.multitask_env.walkers:WalkersOstrichEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id='WalkersKangaroo-v1',
    entry_point='gym_extensions.continuous.mujoco.nervenet_envs.multitask_env.walkers:WalkersKangarooEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)


def get_name_list():
    return name_list


def get_mujoco_model_settings():
    '''
        @brief:
            @traditional environments:
                1. Humanoid-v1
                2. HumanoidStandup-v1
                3. HalfCheetah-v1
                4. Hopper-v1
                5. Walker2d-v1
                6. AntS-v1

            @transfer-learning environments:
                1. Centipede
                2. Snake
                4. Reacher
    '''
    # step 0: settings about the joint
    JOINT_KEY = ['qpos', 'qvel', 'qfrc_constr', 'qfrc_act']
    BODY_KEY = ['cinert', 'cvel', 'cfrc']

    ROOT_OB_SIZE = {
        'qpos': {'free': 7, 'hinge': 1, 'slide': 1},
        'qvel': {'free': 6, 'hinge': 1, 'slide': 1},
        'qfrc_act': {'free': 6, 'hinge': 1, 'slide': 1},
        'qfrc_constr': {'free': 6, 'hinge': 1, 'slide': 1}
    }

    # step 1: register the settings for traditional environments
    SYMMETRY_MAP = {'Humanoid-v1': 2,
                    'HumanoidStandup-v1': 2,
                    'HalfCheetah-v1': 1,
                    'Hopper-v1': 1,
                    'Walker2d-v1': 1,
                    'AntS-v1': 2,
                    'Swimmer-v1': 2,

                    'WalkersHopper-v1': 1,
                    'WalkersHalfhumanoid-v1': 1,
                    'WalkersHalfcheetah-v1': 1,
                    'WalkersFullcheetah-v1': 1,
                    'WalkersOstrich-v1': 1,
                    'WalkersKangaroo-v1': 1}

    XML_DICT = {'Humanoid-v1': 'humanoid.xml',
                'HumanoidStandup-v1': 'humanoid.xml',
                'HalfCheetah-v1': 'half_cheetah.xml',
                'Hopper-v1': 'hopper.xml',
                'Walker2d-v1': 'walker2d.xml',
                'AntS-v1': 'ant.xml',
                'Swimmer-v1': 'SnakeThree.xml',

                'WalkersHopper-v1': 'WalkersHopper.xml',
                'WalkersHalfhumanoid-v1': 'WalkersHalfhumanoid.xml',
                'WalkersHalfcheetah-v1': 'WalkersHalfcheetah.xml',
                'WalkersFullcheetah-v1': 'WalkersFullcheetah.xml',
                'WalkersOstrich-v1': 'WalkersOstrich.xml',
                'WalkersKangaroo-v1': 'WalkersKangaroo.xml'}
    OB_MAP = {
        'Humanoid-v1':
            ['qpos', 'qvel', 'cinert', 'cvel', 'qfrc_act', 'cfrc'],
        'HumanoidStandup-v1':
            ['qpos', 'qvel', 'cinert', 'cvel', 'qfrc_act', 'cfrc'],
        'HalfCheetah-v1': ['qpos', 'qvel'],
        'Hopper-v1': ['qpos', 'qvel'],
        'Walker2d-v1': ['qpos', 'qvel'],
        'AntS-v1': ['qpos', 'qvel', 'cfrc'],
        'Swimmer-v1': ['qpos', 'qvel'],

        'WalkersHopper-v1': ['qpos', 'qvel'],
        'WalkersHalfhumanoid-v1': ['qpos', 'qvel'],
        'WalkersHalfcheetah-v1': ['qpos', 'qvel'],
        'WalkersFullcheetah-v1': ['qpos', 'qvel'],
        'WalkersOstrich-v1': ['qpos', 'qvel'],
        'WalkersKangaroo-v1': ['qpos', 'qvel']
    }

    # step 2: register the settings for the tranfer environments
    SYMMETRY_MAP.update({
        'Centipede': 2,
        'CpCentipede': 2,
        'Snake': 2,
        'Reacher': 0,
    })

    OB_MAP.update({
        'Centipede': ['qpos', 'qvel', 'cfrc'],
        'CpCentipede': ['qpos', 'qvel', 'cfrc'],
        'Snake': ['qpos', 'qvel'],
        'Reacher': ['qpos', 'qvel', 'root_add_5']
    })
    for env in asset_generator.TASK_DICT:
        for i_part in asset_generator.TASK_DICT[env]:
            registered_name = env + num2words.num2words(i_part)[0].upper() \
                + num2words.num2words(i_part)[1:] + '-v1'

            SYMMETRY_MAP[registered_name] = SYMMETRY_MAP[env]
            OB_MAP[registered_name] = OB_MAP[env]
            XML_DICT[registered_name] = registered_name.replace(
                '-v1', '.xml'
            )

    # ob map, symmetry map for robustness task
    for key in ROBUSTNESS_TASK_DICT:
        for env in ROBUSTNESS_TASK_DICT[key]:
            OB_MAP.update({env: ['qpos', 'qvel']})
            SYMMETRY_MAP.update({env: 1})
    # xml dict for botustness task
    for i_agent in range(NUM_ROBUSTNESS_AGENTS + 1):
        XML_DICT.update({
            'WalkersHopper' + num2words.num2words(i_agent) + '-v1':
                'WalkersHopper.xml',
            'WalkersHalfhumanoid' + num2words.num2words(i_agent) + '-v1':
                'WalkersHalfhumanoid.xml',
            'WalkersHalfcheetah' + num2words.num2words(i_agent) + '-v1':
                'WalkersHalfcheetah.xml',
            'WalkersFullcheetah' + num2words.num2words(i_agent) + '-v1':
                'WalkersFullcheetah.xml',
            'WalkersOstrich' + num2words.num2words(i_agent) + '-v1':
                'WalkersOstrich.xml',
        })

    return SYMMETRY_MAP, XML_DICT, OB_MAP, JOINT_KEY, ROOT_OB_SIZE, BODY_KEY
