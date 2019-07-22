# -----------------------------------------------------------------------------
#   @brief:
#       generate the snakes
#   @author:
#       Tingwu Wang, Sept. 1st, 2017
# -----------------------------------------------------------------------------


MUJOCO_XML_HEAD = '''
<mujoco model="swimmer">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option collision="predefined" density="4000" integrator="RK4" timestep="0.01" viscosity="0.1"/>
  <default>
    <geom conaffinity="1" condim="1" contype="1" material="geom" rgba="0.8 0.6 .4 1"/>
    <joint armature='0.1'  />
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="30 30" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
'''

WORLDBODY_XML_HEAD = '''
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1" size="40 40 0.1" type="plane"/>
    <body name="podBody_1" pos="0 0 0">
      <geom name='pod_1' density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
      <joint axis="1 0 0" name="slider1" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" name="slider2" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" name="rot_1" pos="-1.5 0 0" type="hinge"/>
      <site name="tip" pos="0 0 0" size="0.02 0.02"/>
      <body name="podBody_2" pos="-1 0 0">
        <geom name='pod_2' density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="rot_2" pos="0 0 0" range="-100 100" type="hinge"/>
'''

WORLDBODY_XML_TAIL = '''
      </body>
    </body>
  </worldbody>
'''

MUJOCO_XML_TAIL = '''
</mujoco>
'''


def generate_snake(num_pods):
    # add the heads
    xml_content = MUJOCO_XML_HEAD
    xml_content += WORLDBODY_XML_HEAD

    # init the indent level and add two legs
    indent_level = 4
    current_pod_id = 3

    xml_content = _add_body(
        xml_content, current_pod_id, indent_level, num_pods - 2
    )
    xml_content += WORLDBODY_XML_TAIL

    xml_content = _add_actuators(xml_content, num_pods)

    xml_content += MUJOCO_XML_TAIL

    return xml_content


POD_XML_HEAD = '''
<body name="podBody_{POD_ID}" pos="-1 0 0">
  <geom name='pod_{POD_ID}' density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
  <joint axis="0 0 1" limited="true" name="rot_{POD_ID}" pos="0 0 0" range="-100 100" type="hinge"/>
'''


def _add_body(xml_content, current_pod_id, indent_level, num_pods):
    if num_pods - 1 < 0:
        return xml_content

    # add the body head xml
    body_xml_head = POD_XML_HEAD.replace('{POD_ID}', str(current_pod_id))
    body_xml_list = ['  ' * indent_level + lines
                     for lines in body_xml_head.split('\n')]
    xml_content += ('\n'.join(body_xml_list) + '\n')

    # add another layer of body
    xml_content = _add_body(
        xml_content, current_pod_id + 1,
        indent_level + 1, num_pods - 1
    )

    # add the body tail xml
    xml_content += ('  ' * indent_level + '</body>\n')
    return xml_content


def _add_actuators(xml_content, num_pods):
    xml_content += '  <actuator>\n'
    for i_pod in range(2, num_pods + 1):
        xml_content += \
            ('    <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0"' +
             ' joint="rot_{}"/>\n'.format(i_pod))
    xml_content += '  </actuator>\n'
    return xml_content
