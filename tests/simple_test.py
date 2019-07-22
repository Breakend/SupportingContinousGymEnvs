import gym
from gym_extensions.continuous import mujoco

env = gym.make("AntGravityHalf-v1")

env.reset()
for _ in range(1000):
    env.render()
    s, r, d, i = env.step(env.action_space.sample()) # take a random action
    print(s.shape, r, d, i)
env.close()