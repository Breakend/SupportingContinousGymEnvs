# -----------------------------------------------------------------------------
#   @brief:
#       generate the centipedes
#   @author:
#       Tingwu Wang, Sept. 1st, 2017
# -----------------------------------------------------------------------------


import numpy as np
MUJOCO_XML_HEAD = '''
<mujoco model="centipede">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>

  <option integrator="RK4" timestep="0.01"/>

  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="25.0" friction="1.5 0.1 0.1" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>

  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
'''

WORLDBODY_XML_HEAD = '''
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso_0" pos="0 0 0.75">
      <geom name="torsoGeom_0" pos="0 0 0" size="0.25" type="sphere" density="100"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
'''

WORLDBODY_XML_TAIL = '''
    </body>
  </worldbody>
'''

MUJOCO_XML_TAIL = '''
</mujoco>
'''


def generate_centipede(num_legs):
    # add the heads
    xml_content = MUJOCO_XML_HEAD
    xml_content += WORLDBODY_XML_HEAD

    # init the indent level and add two legs
    indent_level = 3
    current_leg_id = 0
    current_body_id = 1
    xml_content, current_leg_id = _add_leg(xml_content, current_leg_id,
                                           indent_level)
    xml_content, current_leg_id = _add_leg(xml_content, current_leg_id,
                                           indent_level)

    xml_content = _add_body(
        xml_content, current_leg_id, current_body_id,
        indent_level, num_legs - 2
    )
    xml_content += WORLDBODY_XML_TAIL

    xml_content = _add_actuators(xml_content, num_legs)

    xml_content = _add_custom(xml_content, num_legs)

    xml_content += MUJOCO_XML_TAIL
    return xml_content


LEG_XML = '''
<body name="legbody_{LEG_ID}" pos="0.0 {LEG_LEN_2} 0">
  <joint axis="0 0 1" name="hip_{LEG_ID}" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
  <geom fromto="0.0 0.0 0.0 0.0 {LEG_LEN_2} 0.0" name="legGeom_{LEG_ID}" size="0.08" type="capsule" rgba=".08 .5 .3 1"/>
  <body pos="0 {LEG_LEN_2} 0" name="frontFoot_{LEG_ID}">
    <joint axis="{AXIS} 0 0" name="ankle_{LEG_ID}" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
    <geom fromto="0.0 0.0 0.0 0 {LEG_LEN_3} 0.0" name="ankleGeom_{LEG_ID}" size="0.08" type="capsule" rgba=".08 .5 .3 1"/>
  </body>
</body>
'''

BODY_XML_HEAD = '''
<body name="torso_{BODY_ID}" pos="0.50 0 0">
  <geom name="torsoGeom_{BODY_ID}" pos="0 0 0" size="0.25" type="sphere" density="100"/>
  <joint axis="0 0 1" name="body_{BODY_ID}" pos="-0.25 0.0 0.0" range="-20 20" type="hinge"/>
  <joint axis="0 1 0" name="bodyupdown_{BODY_ID}" pos="-0.25 0.0 0.0" range="-10 30" type="hinge"/>
'''
LEG_PARA = {
    '{LEG_LEN_1}': 0.25,
    '{LEG_LEN_2}': 0.28,
    '{LEG_LEN_3}': 0.6,
    '{AXIS}': -1
}


def _add_leg(xml_content, current_leg_id, indent_level):
    '''
        @brief: add one leg to the model
    '''
    is_right_leg = (current_leg_id % 2) * 2 - 1
    xml_to_add = LEG_XML.replace('{LEG_ID}', str(current_leg_id))
    # NOTE: break the left right symmetry here!
    if is_right_leg > 0:
        xml_to_add = xml_to_add.replace('hip_', 'righthip_')
    else:
        xml_to_add = xml_to_add.replace('hip_', 'lefthip_')
    for para in LEG_PARA:
        xml_to_add = xml_to_add.replace(para,
                                        str(LEG_PARA[para] * is_right_leg))

    xml_list = ['  ' * indent_level + lines
                for lines in xml_to_add.split('\n')]
    xml_content += ('\n'.join(xml_list) + '\n')

    return xml_content, current_leg_id + 1


def _add_body(xml_content, current_leg_id, current_body_id,
              indent_level, num_legs):
    if num_legs <= 0:
        return xml_content
    else:

        # add the body head xml
        body_xml_head = BODY_XML_HEAD.replace('{BODY_ID}', str(current_body_id))
        body_xml_list = ['  ' * indent_level + lines
                         for lines in body_xml_head.split('\n')]
        xml_content += ('\n'.join(body_xml_list) + '\n')

        # add two legs or one legs
        if num_legs >= 2:
            xml_content, current_leg_id = _add_leg(xml_content, current_leg_id,
                                                   indent_level + 1)
            xml_content, current_leg_id = _add_leg(xml_content, current_leg_id,
                                                   indent_level + 1)
        else:
            xml_content, current_leg_id = _add_leg(xml_content, current_leg_id,
                                                   indent_level + 1)
        # add another layer of body
        xml_content = _add_body(
            xml_content, current_leg_id,
            current_body_id + 1, indent_level + 1,
            num_legs - 2
        )

        # add the body tail xml
        xml_content += ('  ' * indent_level + '</body>\n')
        return xml_content


def _add_actuators(xml_content, num_legs):
    xml_content += '  <actuator>\n'
    for i_leg in range(num_legs):
        is_right_leg = (i_leg % 2) * 2 - 1
        if is_right_leg > 0:
            xml_content += \
                ('    <motor ctrllimited="true" ctrlrange="-1.0 1.0"' +
                 ' joint="righthip_{}" gear="150"/>\n'.format(i_leg))
        else:
            xml_content += \
                ('    <motor ctrllimited="true" ctrlrange="-1.0 1.0"' +
                 ' joint="lefthip_{}" gear="150"/>\n'.format(i_leg))
        xml_content += \
            ('    <motor ctrllimited="true" ctrlrange="-1.0 1.0"' +
             ' joint="ankle_{}" gear="150"/>\n'.format(i_leg))
        if i_leg % 2 == 1 and i_leg > 1:
            # add the body joint
            xml_content += \
                ('    <motor ctrllimited="true" ctrlrange="-1.0 1.0"' +
                 ' joint="body_{}" gear="100"/>\n'.format(
                     int(np.floor(i_leg / 2))))
            xml_content += \
                ('    <motor ctrllimited="true" ctrlrange="-1.0 1.0"' +
                 ' joint="bodyupdown_{}" gear="100"/>\n'.format(
                     int(np.floor(i_leg / 2))))

    xml_content += '  </actuator>\n'
    return xml_content


def _add_custom(xml_content, num_legs):
    xml_content += '  <custom>\n'
    xml_content += '    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0'

    for i_leg in range(num_legs):
        xml_content += ' ' + str(0.0) + ' ' + str(1.0 * ((i_leg % 2) * 2 - 1))

    xml_content += '" name="init_qpos"/>\n'

    xml_content += '  </custom>\n'
    return xml_content
