import gymnasium as gym
from gymnasium import envs

# List all classic control environments
classic_envs = [
    "CartPole-v0", "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "Acrobot-v1"
]

print("Classic Control Environments:")
for env_name in classic_envs:
    env = gym.make(env_name)
    print(f"\n{env_name}:")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    env.close()