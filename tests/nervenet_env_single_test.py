import gym
import gym_extensions.continuous.mujoco.nervenet_envs.register as register

env = gym.make("WalkersHopperthree-v1")
env.reset()

for _ in range(100*3):
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    # print(state.shape, action.shape, reward, done, info)