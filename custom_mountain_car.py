import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

class CustomMountainCar(MountainCarEnv):
    def __init__(
        self, 
        render_mode=None,
        # Physics parameters with defaults
        min_position=-1.2,
        max_position=0.6,
        force=0.001,
        gravity=0.0025,
        max_speed=0.07,
        goal_position=0.5,
        # Initial state parameters
        init_position_low=-0.6,
        init_position_high=-0.4
    ):
        # Initialize with render_mode
        super().__init__(render_mode=render_mode)
        
        # Set customizable physics parameters
        self.min_position = min_position
        self.max_position = max_position
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed
        self.goal_position = goal_position
        
        # Store initial position ranges
        self.init_position_low = init_position_low
        self.init_position_high = init_position_high
        
        # Update observation space to match new bounds
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)
    
    def reset(self, **kwargs):
        # Override parent reset to use our custom initial position range
        super().reset(seed=kwargs.get('seed', None))
        
        options = kwargs.get('options', {})
        if options is not None and 'x_init' in options and 'y_init' in options:
            # Use provided options
            self.state = np.array([options['x_init'], options['y_init']])
        else:
            # Use our custom initialization ranges
            self.state = np.array([
                self.np_random.uniform(low=self.init_position_low, high=self.init_position_high),
                0.0  # Initial velocity is always 0
            ])
            
        if self.render_mode == "human":
            self.render()
        
        return np.array(self.state, dtype=np.float32), {}
    
    def step(self, action):
        # Call parent step method but with our custom physics
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(position >= self.goal_position and velocity >= 0)
        
        # Base reward
        reward = -1.0
        
        # Additional reward components
        height = self._height(position)
        height_reward = height * 0.1
        
        velocity_reward = 0
        if position < self.goal_position:
            velocity_reward = velocity * 0.1
        
        modified_reward = reward + height_reward + velocity_reward
        
        self.state = (position, velocity)
        
        # Add reward details to info
        info = {
            'height_reward': height_reward,
            'velocity_reward': velocity_reward,
            'base_reward': reward
        }
        
        if self.render_mode == "human":
            self.render()
            
        return np.array(self.state, dtype=np.float32), modified_reward, terminated, False, info


# Test with different physics configurations
if __name__ == "__main__":
    print("\n*** Testing Easy Mountain Car Environment ***")
    # Create an easier version of the environment
    easy_env = CustomMountainCar(
        render_mode="human",
        min_position=-1.5,
        force=0.003,         # Stronger engine
        gravity=0.0015,      # Less gravity
        max_speed=0.1,       # Higher max speed
        goal_position=0.45,  # Easier goal
        init_position_low=-0.6,
        init_position_high=-0.4
    )
    
    state, info = easy_env.reset(seed=42)
    print(f"Easy Environment Parameters:")
    print(f"  Force: {easy_env.force} - Stronger engine")
    print(f"  Gravity: {easy_env.gravity} - Less gravity")
    print(f"  Max Speed: {easy_env.max_speed} - Higher top speed")
    print(f"  Goal Position: {easy_env.goal_position} - Easier goal")
    
    # Test for a few steps
    for i in range(200):
        action = easy_env.action_space.sample()
        state, reward, terminated, truncated, info = easy_env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: Position = {state[0]:.2f}, Velocity = {state[1]:.2f}, Reward = {reward:.4f}")
        
        if terminated:
            print(f"Easy environment - Goal reached at step {i}!")
            break
    
    easy_env.close()
    
    # Wait for user input before starting the hard environment
    input("\nPress Enter to continue to the Hard Environment...\n")
    
    print("*** Testing Hard Mountain Car Environment ***")
    # Create a harder version of the environment
    hard_env = CustomMountainCar(
        render_mode="human",
        min_position=-2.0,
        force=0.0008,        # Weaker engine
        gravity=0.003,       # More gravity
        max_speed=0.06,      # Lower max speed
        goal_position=0.55,  # Harder goal
        init_position_low=-1.1,
        init_position_high=-0.9
    )
    
    state, info = hard_env.reset(seed=42)
    print(f"Hard Environment Parameters:")
    print(f"  Force: {hard_env.force} - Weaker engine")
    print(f"  Gravity: {hard_env.gravity} - More gravity")
    print(f"  Max Speed: {hard_env.max_speed} - Lower top speed")
    print(f"  Goal Position: {hard_env.goal_position} - Harder goal")
    print(f"  Starting position range: [{hard_env.init_position_low}, {hard_env.init_position_high}]")
    
    # Test for a few steps
    for i in range(300):
        action = hard_env.action_space.sample()
        state, reward, terminated, truncated, info = hard_env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: Position = {state[0]:.2f}, Velocity = {state[1]:.2f}, Reward = {reward:.4f}")
        
        if terminated:
            print(f"Hard environment - Goal reached at step {i}!")
            break
    
    hard_env.close()