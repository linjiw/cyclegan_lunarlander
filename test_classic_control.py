import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
env.reset()
for _ in range(100):
    action = env.action_space.sample()
    env.step(action)
env.close()
