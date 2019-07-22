# -----------------------------------------------------------------------------
#   @brief:
#       generate the xml files for each different sub-tasks of one master task
#   @author:
#       Tingwu Wang, Aug. 30th, 2017
# -----------------------------------------------------------------------------

import argparse
import gym_extensions.continuous.mujoco.nervenet_envs.init_path as init_path
import os
import num2words
import gym_extensions.continuous.mujoco.nervenet_envs.centipede_generator as centipede_generator
import gym_extensions.continuous.mujoco.nervenet_envs.snake_generator as snake_generator
import gym_extensions.continuous.mujoco.nervenet_envs.reacher_generator as reacher_generator

TASK_DICT = {
    'Centipede': [3, 5, 7] + [4, 6, 8, 10, 12, 14] + [20, 30, 40, 50],
    # 'CpCentipede': [3, 5, 7] + [4, 6, 8, 10, 12, 14], # this doesn't exist anymore
    'Reacher': [0, 1, 2, 3, 4, 5, 6, 7],
    'Snake': [3, 4, 5, 6, 7, 8, 9] + [10, 20, 40],
}
OUTPUT_BASE_DIR = os.path.join(init_path.get_abs_base_dir(),
                               'environments', 'assets')


def save_xml_files(model_names, xml_number, xml_contents):
    # get the xml path ready
    number_str = num2words.num2words(xml_number)
    xml_names = model_names + number_str[0].upper() + number_str[1:] + '.xml'
    xml_file_path = os.path.join(OUTPUT_BASE_DIR, xml_names)

    # save the xml file
    f = open(xml_file_path, 'w')
    f.write(xml_contents)
    f.close()


GENERATOR_DICT = {
    'Centipede': centipede_generator.generate_centipede,
    'Snake': snake_generator.generate_snake,
    'Reacher': reacher_generator.generate_reacher
}

if __name__ == '__main__':
    # parse the parameters
    parser = argparse.ArgumentParser(description='xml_asset_generator.')
    parser.add_argument("--env_name", type=str, default='Centipede')
    args = parser.parse_args()

    # generator the environment xmls
    for i_leg_num in TASK_DICT[args.env_name]:
        xml_contents = GENERATOR_DICT[args.env_name](i_leg_num)
        save_xml_files(args.env_name, i_leg_num, xml_contents)
