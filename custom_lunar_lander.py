import math
import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander, FPS, VIEWPORT_W, VIEWPORT_H, SCALE
from gymnasium.utils import EzPickle

class LeftWindLunarLander(LunarLander, EzPickle):
    """
    Lunar Lander with strong consistent wind from the right (pushing lander to the left).
    This makes the landing challenge harder as the lander must compensate for the leftward push.
    """
    
    def __init__(
        self,
        render_mode=None,
        continuous=False,
        gravity=-10.0,
    ):
        EzPickle.__init__(self, render_mode, continuous, gravity)
        super().__init__(
            render_mode=render_mode,
            continuous=continuous,
            gravity=gravity,
            enable_wind=True,
            wind_power=20.0,  # Maximum recommended wind power
            turbulence_power=1.5,
        )
    
    def step(self, action):
        # Apply left-directed wind (from right to left)
        if not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # Override the default wind pattern with constant leftward force
            # Always apply negative wind force (pushing to the left)
            self.lander.ApplyForceToCenter((-self.wind_power, 0.0), True)
            
            # Apply a slight upward force to simulate complex wind patterns
            self.lander.ApplyForceToCenter((0.0, self.turbulence_power), True)
            
            # Apply some random torque for realism (makes it harder)
            torque_mag = self.np_random.uniform(-0.5, 0.5) * self.turbulence_power
            self.lander.ApplyTorque(torque_mag, True)
        
        # Continue with the normal step logic (actions, rewards, etc.)
        return super().step(action)


class RightWindLunarLander(LunarLander, EzPickle):
    """
    Lunar Lander with strong consistent wind from the left (pushing lander to the right).
    This makes the landing challenge harder as the lander must compensate for the rightward push.
    """
    
    def __init__(
        self,
        render_mode=None,
        continuous=False,
        gravity=-10.0,
    ):
        EzPickle.__init__(self, render_mode, continuous, gravity)
        super().__init__(
            render_mode=render_mode,
            continuous=continuous,
            gravity=gravity,
            enable_wind=True,
            wind_power=20.0,  # Maximum recommended wind power
            turbulence_power=1.5,
        )
    
    def step(self, action):
        # Apply right-directed wind (from left to right)
        if not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # Override the default wind pattern with constant rightward force
            # Always apply positive wind force (pushing to the right)
            self.lander.ApplyForceToCenter((self.wind_power, 0.0), True)
            
            # Apply a slight downward force to make it harder than the left wind version
            self.lander.ApplyForceToCenter((0.0, -self.turbulence_power), True)
            
            # Apply some random torque for realism (makes it harder)
            torque_mag = self.np_random.uniform(-0.5, 0.5) * self.turbulence_power
            self.lander.ApplyTorque(torque_mag, True)
        
        # Continue with the normal step logic (actions, rewards, etc.)
        return super().step(action)


# Register the environments
gym.register(
    id="LunarLanderLeftWind-v0",
    entry_point=LeftWindLunarLander,
    max_episode_steps=1000,
    reward_threshold=200,
)

gym.register(
    id="LunarLanderRightWind-v0",
    entry_point=RightWindLunarLander,
    max_episode_steps=1000,
    reward_threshold=200,
)

# Testing code
if __name__ == "__main__":
    import time
    
    # Test Left Wind Environment
    print("\n*** Testing Left Wind Lunar Lander Environment ***")
    left_env = gym.make("LunarLanderLeftWind-v0", render_mode="human")
    left_env.reset(seed=42)
    
    # Access the underlying environment with unwrapped
    base_left_env = left_env.unwrapped
    
    print("Left Wind Environment: Consistent strong wind pushing left")
    print(f"  Wind Power: {base_left_env.wind_power}")
    print(f"  Turbulence: {base_left_env.turbulence_power}")
    print(f"  Gravity: {base_left_env.gravity}")
    
    steps = 0
    total_reward = 0
    
    try:
        while True:
            # Sample a random action
            action = left_env.action_space.sample()
            state, reward, terminated, truncated, info = left_env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 50 == 0:
                print(f"Step {steps}: Total Reward = {total_reward:.2f}")
            
            if terminated or truncated:
                print(f"Left wind environment - Episode ended after {steps} steps with reward {total_reward:.2f}")
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    left_env.close()
    
    # Wait before starting the next environment
    time.sleep(1)
    
    # Test Right Wind Environment
    print("\n*** Testing Right Wind Lunar Lander Environment ***")
    right_env = gym.make("LunarLanderRightWind-v0", render_mode="human")
    right_env.reset(seed=42)
    
    # Access the underlying environment with unwrapped
    base_right_env = right_env.unwrapped
    
    print("Right Wind Environment: Consistent strong wind pushing right")
    print(f"  Wind Power: {base_right_env.wind_power}")
    print(f"  Turbulence: {base_right_env.turbulence_power}")
    print(f"  Gravity: {base_right_env.gravity}")
    
    steps = 0
    total_reward = 0
    
    try:
        while True:
            # Sample a random action
            action = right_env.action_space.sample()
            state, reward, terminated, truncated, info = right_env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 50 == 0:
                print(f"Step {steps}: Total Reward = {total_reward:.2f}")
            
            if terminated or truncated:
                print(f"Right wind environment - Episode ended after {steps} steps with reward {total_reward:.2f}")
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    right_env.close()