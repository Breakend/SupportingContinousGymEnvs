# -----------------------------------------------------------------------------
#   @brief:
#       generate the reacher
#   @author:
#       Tingwu Wang, Sept. 3rd, 2017
# -----------------------------------------------------------------------------


MUJOCO_XML_HEAD = '''
<mujoco model="modified_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
'''

WORLDBODY_XML_HEAD = '''
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>
    <geom conaffinity="0" fromto="-{RANGE} -{RANGE} .01 {RANGE} -{RANGE} .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".01" type="capsule"/>
    <geom conaffinity="0" fromto=" {RANGE} -{RANGE} .01 {RANGE} {RANGE} .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".01" type="capsule"/>
    <geom conaffinity="0" fromto="-{RANGE} {RANGE} .01 {RANGE} {RANGE} .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".01" type="capsule"/>
    <geom conaffinity="0" fromto="-{RANGE} -{RANGE} .01 -{RANGE} {RANGE} .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".01" type="capsule"/>

    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="rootBody" pos="0 0 .01">
      <geom fromto="0 0 0 0.1 0 0" name="rootPod" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
      <joint axis="0 0 1" limited="true" name="rootjoint" pos="0 0 0" range='-3.14 3.14' type="hinge"/>
'''

WORLDBODY_XML_TAIL = '''
    </body>
    <body name="target" pos=".1 -.1 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="targetX" pos="0 0 0" range="-{RANGE} {RANGE}" ref=".1" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="targetY" pos="0 0 0" range="-{RANGE} {RANGE}" ref="-.1" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
    </body>
    <body name="indicator" pos="0 0 .01">
      <geom conaffinity="0" contype="0" name="reacherIndicator" pos="0 0 0" rgba="0 0.9 0 1" size=".001" type="sphere"/>
      <geom conaffinity="0" contype="0" name="avoiderIndicator" pos="0 0 0" rgba="0.9 0 0 1" size=".001" type="sphere"/>
    </body>
  </worldbody>
'''

MUJOCO_XML_TAIL = '''
</mujoco>
'''


def generate_reacher(num_pods):
    # add the heads
    xml_content = MUJOCO_XML_HEAD
    xml_content += WORLDBODY_XML_HEAD.replace('{RANGE}', str(num_pods * 0.1 + 0.15))

    # init the indent level and add two legs
    indent_level = 3
    current_pod_id = 1

    xml_content = _add_body(
        xml_content, current_pod_id, indent_level, num_pods
    )
    xml_content += WORLDBODY_XML_TAIL.replace('{RANGE}', str(num_pods * 0.1 + 0.1))

    xml_content = _add_actuators(xml_content, num_pods + 1)

    xml_content += MUJOCO_XML_TAIL

    return xml_content


POD_XML_HEAD = '''
<body name="body_{POD_ID}" pos="0.1 0 0">
  <joint axis="0 0 1" limited="true" name="joint_{POD_ID}" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
  <geom fromto="0 0 0 0.1 0 0" name="link_{POD_ID}" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
'''
FINGER_TIP_XML = '''
<geom contype="0" name="fingertip" pos="0.11 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
<site name="tip" pos="0.11 0 0" size="0.001 0.001"/>
'''


def _add_body(xml_content, current_pod_id, indent_level, num_pods):
    if num_pods - 1 < 0:
        body_xml_list = ['  ' * indent_level + lines
                         for lines in FINGER_TIP_XML.split('\n')]
        return xml_content + ('\n'.join(body_xml_list) + '\n')

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
    xml_content += ('    <motor ctrllimited="true" ctrlrange="-1 1"' +
                    ' gear="200.0" joint="rootjoint"/>\n')
    for i_pod in range(1, num_pods):
        xml_content += \
            ('    <motor ctrllimited="true" ctrlrange="-1 1" gear="200.0"' +
             ' joint="joint_{}"/>\n'.format(i_pod))
    xml_content += '  </actuator>\n'
    return xml_content
