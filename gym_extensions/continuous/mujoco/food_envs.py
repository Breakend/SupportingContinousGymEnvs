import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import random
from gym import utils
import math, pyrr
from gym_extensions.continuous.mujoco.wall_envs import isclose, rotate_vector

FOOD_BONUS = 100


def SingleFoodEnvFactory(class_type):
    """class_type should be an OpenAI gym time"""

    class SingleFoodEnv(class_type, utils.EzPickle):
        """
        This provides a piece of frictionless food, which the agent must try to seek in the env.
        we might be using visual observation instead of sensors installed.
        """

        def __init__(
                self,
                model_path,
                food_height=.25,
                food_pos_range=([1.5, 1.5], [2.5, 2.5]),
                # food_pos_range=([0.5, 0.5], [0.8, 0.8]), # test purpose
                n_bins=10,
                sensor_range=10.,
                sensor_span=math.pi / 2,
                *args,
                **kwargs):

            # TODO: temp workaround: seems openai gym requires this tho, I don't better way to specify this.
            self._seed = 0

            self._n_bins = n_bins
            # Add a sensor
            self._sensor_range = sensor_range
            self._sensor_span = sensor_span

            tree = ET.parse(model_path)
            worldbody = tree.find(".//worldbody")

            height = food_height
            self.food_pos_range = food_pos_range
            rand_x = random.uniform(food_pos_range[0][0], food_pos_range[1][0])
            rand_y = random.uniform(food_pos_range[0][1], food_pos_range[1][1])
            self.food_pos = food_pos = (rand_x, rand_y)
            torso_x, torso_y = 0, 0
            self._init_torso_x = torso_x
            self.class_type = class_type
            self._init_torso_y = torso_y
            self.food_size = (0.2, 0.4, height)

            ET.SubElement(
                worldbody, "geom",
                name="food",
                pos="%f %f %f" % (food_pos[0],
                                  food_pos[1],
                                  height / 2.),
                size="%f %f %f" % self.food_size,
                type="sphere",
                material="",
                contype="1",
                conaffinity="1",
                density="0.00001",
                rgba="1.0 0. 1. 1"
            )

            torso = tree.find(".//body[@name='torso']")
            geoms = torso.findall(".//geom")
            for geom in geoms:
                if 'name' not in geom.attrib:
                    raise Exception("Every geom of the torso must have a name "
                                    "defined")

            # MuJoCo200 only accepts the file extension of "xml"
            _, file_path = tempfile.mkstemp(suffix=".xml", text=True)
            tree.write(file_path)

            # self._goal_range = self._find_goal_range()
            self._cached_segments = None

            class_type.__init__(self, model_path=file_path)
            utils.EzPickle.__init__(self)

            # import pdb; pdb.set_trace()

        def get_body_xquat(self, body_name):
            idx = self.model.body_names.index(body_name)
            return self.sim.data.body_xquat[idx]

        def reset(self):
            temp = np.copy(self.model.geom_pos)

            rand_x = random.uniform(self.food_pos_range[0][0], self.food_pos_range[1][0])
            rand_y = random.uniform(self.food_pos_range[0][1], self.food_pos_range[1][1])

            # TODO: make this more robust,
            # hardcoding that the second geom is the food,
            # but we should do something more robust??
            assert isclose(temp[1][0], self.food_pos[0])
            assert isclose(temp[1][1], self.food_pos[1])

            self.food_pos = (rand_x, rand_y)
            self.model.geom_pos[1][0] = self.food_pos[0]
            self.model.geom_pos[1][1] = self.food_pos[1]
            ob = super(SingleFoodEnv, self).reset()
            return ob

        def _get_obs(self):
            """
            The observation would include both information about
            the robot itself as well as the sensors around its environment
            """
            robot_x, robot_y, robot_z = robot_coords = self.get_body_com("torso")
            food_readings = np.zeros(self._n_bins)

            for ray_idx in range(self._n_bins):
                # self._sensor_span * 0.5 + 1.0 * (2 * ray_idx + 1) / (2 * self._n_bins) * self._sensor_span
                theta = (self._sensor_span / self._n_bins) * ray_idx - self._sensor_span / 2.
                forward_normal = rotate_vector(np.array([1, 0, 0]), [0, 1, 0], theta)
                # Note: Mujoco quaternions use [w, x, y, z] convention
                quat_mujoco = self.get_body_xquat("torso")
                quat = [quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]]
                ray_direction = pyrr.quaternion.apply_to_vector(quat, forward_normal)
                ray = pyrr.ray.create(robot_coords, ray_direction)

                bottom_point = [self.food_pos[0] - self.food_size[0] / 2.,
                                self.food_pos[1] - self.food_size[1] / 2.,
                                0.]
                top_point = [self.food_pos[0] + self.food_size[0] / 2.,
                             self.food_pos[1] + self.food_size[1] / 2.,
                             self.food_size[2]]

                # import pdb; pdb.set_trace()
                bounding_box = pyrr.aabb.create_from_points([bottom_point, top_point])
                intersection = pyrr.geometric_tests.ray_intersect_aabb(ray, bounding_box)

                if intersection is not None:
                    distance = np.linalg.norm(intersection - robot_coords)
                    if distance <= self._sensor_range:
                        food_readings[ray_idx] = distance / self._sensor_range

            obs = np.concatenate([
                self.class_type._get_obs(self),
                food_readings
            ])

            return obs

        def _is_in_collision(self, pos):
            x, y = pos

            minx = self.food_pos[0] * 1 - 1 * 0.5 - self._init_torso_x
            maxx = self.food_pos[0] * 1 + 1 * 0.5 - self._init_torso_x
            miny = self.food_pos[1] * 1 - 1 * 0.5 - self._init_torso_y
            maxy = self.food_pos[1] * 1 + 1 * 0.5 - self._init_torso_y
            if minx <= x <= maxx and miny <= y <= maxy:
                return True
            return False

        def _get_food_bonus(self):
            robot_x, robot_y, robot_z = self.get_body_com("torso")
            if self._is_in_collision((robot_x, robot_y)):
                return FOOD_BONUS
            else:
                return 0

        def _is_done(self):
            robot_x, robot_y, robot_z = self.get_body_com("torso")
            if self._is_in_collision((robot_x, robot_y)):
                return True
            else:
                return False

        def get_xy(self):
            return self.get_body_com("torso")[:2]

        def step(self, action):
            state, reward, done, info = super(SingleFoodEnv, self).step(action)

            next_obs = self._get_obs()
            reward += self._get_food_bonus()  # TODO: think about the reward func design
            done = self._is_done()

            x, y = self.get_body_com("torso")[:2]
            return next_obs, reward, done, info

        def action_from_key(self, key):
            return self.action_from_key(key)

    return SingleFoodEnv

