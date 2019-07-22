import gym
from gym_extensions.continuous import mujoco

env = gym.make("AntSingleFood-v1")

env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    s, r, d, i = env.step(action) # take a random action
    print(action, s.shape, r, d, i)
    if d:
        break
env.close()