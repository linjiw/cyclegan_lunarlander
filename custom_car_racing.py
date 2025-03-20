import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing, TRACK_WIDTH, PLAYFIELD, FPS
from gymnasium.envs.box2d.car_dynamics import Car

class CustomCarRacing(CarRacing):
    def __init__(
        self,
        render_mode=None,
        # Customizable track parameters
        track_width=40.0/6.0,  # Default matches original (40/SCALE)
        track_length_factor=1.0,
        playfield_size=2000.0/6.0,  # Default matches original (2000/SCALE)
        # Customizable reward parameters
        tile_visited_reward=1000.0,
        frame_cost=0.1,
        out_of_bounds_penalty=100,
        # Customizable car physics parameters
        friction=1.0,
        engine_power=100.0,
        max_steering=1.5,
        max_speed=None,
        # Additional parameters
        lap_complete_percent=0.95,
        domain_randomize=False, 
        continuous=True
    ):
        # Initialize with parent constructor parameters
        super().__init__(
            render_mode=render_mode,
            lap_complete_percent=lap_complete_percent,
            domain_randomize=domain_randomize,
            continuous=continuous
        )
        
        # Override track parameters
        self.custom_track_width = track_width  # Store our custom track width
        self.track_length_factor = track_length_factor
        self.custom_playfield = playfield_size
        
        # Store car physics parameters for later use when creating the car
        self.friction = friction
        self.engine_power = engine_power
        self.max_steering = max_steering
        self.max_speed = max_speed
        
        # Reward parameters
        self.tile_visited_reward = tile_visited_reward
        self.frame_cost = frame_cost
        self.out_of_bounds_penalty = out_of_bounds_penalty
        
    def _create_track(self):
        # Override the global TRACK_WIDTH with our custom width
        global TRACK_WIDTH
        original_track_width = TRACK_WIDTH
        TRACK_WIDTH = self.custom_track_width
        
        # Override the global PLAYFIELD with our custom size
        global PLAYFIELD
        original_playfield = PLAYFIELD
        PLAYFIELD = self.custom_playfield
        
        # Modify the number of checkpoints based on track length factor
        CHECKPOINTS = 12
        checkpoints = int(CHECKPOINTS * self.track_length_factor)
        checkpoints = max(checkpoints, 6)  # Ensure minimum viable track
        
        # Call parent method with modified globals
        success = super()._create_track()
        
        # Restore the original globals (important to avoid side effects)
        TRACK_WIDTH = original_track_width
        PLAYFIELD = original_playfield
        
        return success
    
    def reset(self, **kwargs):
        # Call parent reset
        observation, info = super().reset(**kwargs)
        
        # Now we need to replace the default car with our custom car
        if self.car:
            # Destroy the default car created by parent
            self.car.destroy()
            
            # Create a new car with our custom physics parameters
            self.car = CustomCar(
                self.world, 
                *self.track[0][1:4],
                friction=self.friction,
                engine_power=self.engine_power,
                max_steering=self.max_steering,
                max_speed=self.max_speed
            )
        
        return observation, info
    
    def step(self, action):
        # Call parent step but modify the reward structure
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Replace the parent reward with our custom reward structure
        # Base frame penalty
        modified_reward = -self.frame_cost
        
        # Reward for visited tiles
        if self.tile_visited_count > 0:
            modified_reward += self.tile_visited_reward / len(self.track) * self.tile_visited_count
        
        # Apply the out-of-bounds penalty if terminated due to going off track
        if terminated and not getattr(self, 'lap_complete', False):
            modified_reward -= self.out_of_bounds_penalty
        
        # Add detailed reward info
        info['frame_cost'] = -self.frame_cost
        info['tile_reward'] = (self.tile_visited_reward / len(self.track)) * self.tile_visited_count if self.tile_visited_count > 0 else 0.0
        
        return observation, modified_reward, terminated, truncated, info


# Custom Car class that extends the original Car to allow physics customization
class CustomCar(Car):
    def __init__(self, world, init_angle, init_x, init_y, friction=1.0, engine_power=100.0, max_steering=1.5, max_speed=None):
        # Initialize the parent Car
        super().__init__(world, init_angle, init_x, init_y)
        
        # Store customized parameters
        self.custom_engine_power = engine_power / 100.0
        self.custom_friction = friction
        self.max_steer = max_steering  # This affects the steering angle limit
        self.max_speed = max_speed
        
        # In the original Car implementation, these values are used in step() method
        # We'll override the methods that use these values instead of trying to modify
        # the Box2D objects directly
    
    def gas(self, gas):
        """Apply gas with customized engine power"""
        # The original implementation applies gas to the back wheels
        # We'll modify the gas value by our engine power multiplier
        if not gas:
            return
        
        effective_gas = gas * self.custom_engine_power
        
        # The rest is similar to the parent implementation but with our scaling
        for w in [self.wheels[2], self.wheels[3]]:
            diff = effective_gas - w.omega
            if diff > 0.1:
                diff = 0.1
            if diff < -0.1:
                diff = -0.1
            w.gas_force = 1000000 * diff * effective_gas
    
    def brake(self, b):
        """Apply brake with customized friction"""
        # Scale the braking force by our friction multiplier
        effective_brake = b * self.custom_friction
        
        # Apply to all wheels (similar to parent but with our scaling)
        for w in self.wheels:
            w.brake_force = 15 * effective_brake
    
    def step(self, dt):
        # Call parent step method first
        super().step(dt)
        
        # Apply max speed limitation if set
        if self.max_speed is not None:
            velocity = np.sqrt(
                np.square(self.hull.linearVelocity[0]) + 
                np.square(self.hull.linearVelocity[1])
            )
            if velocity > self.max_speed:
                # Scale down velocity to max_speed
                scale = self.max_speed / velocity
                self.hull.linearVelocity[0] *= scale
                self.hull.linearVelocity[1] *= scale


# Test with different configurations
if __name__ == "__main__":
    print("\n*** Testing Easy Car Racing Environment ***")
    # Create an easier version of the environment
    easy_env = CustomCarRacing(
        render_mode="human",
        track_width=10.0,         # Wider track (original is 40/6 ≈ 6.67)
        track_length_factor=0.8,  # Shorter track
        tile_visited_reward=1200, # More reward
        frame_cost=0.05,          # Less penalty per frame
        engine_power=130.0,       # Stronger engine (130% of normal)
        friction=0.9,             # Less friction (easier to drive)
        max_steering=1.8,         # More responsive steering
        continuous=False          # Discrete actions for easier control
    )
    
    state, info = easy_env.reset(seed=42)
    print(f"Easy Environment Parameters:")
    print(f"  Track Width: {easy_env.custom_track_width} - Wider track")
    print(f"  Track Length Factor: {easy_env.track_length_factor} - Shorter track")
    print(f"  Tile Reward: {easy_env.tile_visited_reward} - More reward")
    print(f"  Frame Cost: {easy_env.frame_cost} - Less penalty")
    print(f"  Engine Power: {easy_env.engine_power} - Stronger engine")
    print(f"  Friction: {easy_env.friction} - Less friction")
    
    # Let the user play or run a simple policy
    total_reward = 0
    steps = 0
    
    # Simple "play" loop - either run with keyboard input or with a simple policy
    try:
        while True:
            # For automatic testing, sample random actions
            action = easy_env.action_space.sample()
            
            state, reward, terminated, truncated, info = easy_env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"Step {steps}: Total Reward = {total_reward:.2f}")
            
            if terminated or truncated:
                print(f"Easy environment - Episode ended after {steps} steps with reward {total_reward:.2f}")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    easy_env.close()
    
    # Wait for user input before starting the hard environment
    input("\nPress Enter to continue to the Hard Environment...\n")
    
    print("*** Testing Hard Car Racing Environment ***")
    # Create a harder version of the environment
    hard_env = CustomCarRacing(
        render_mode="human",
        track_width=4.0,          # Narrower track
        track_length_factor=1.3,  # Longer track
        playfield_size=350,       # Smaller playfield (easier to go out of bounds)
        tile_visited_reward=800,  # Less reward
        frame_cost=0.15,          # More penalty per frame
        out_of_bounds_penalty=150,# Bigger penalty for going off-track
        engine_power=80.0,        # Weaker engine (80% of normal)
        friction=1.2,             # More friction (harder to control)
        max_steering=1.2,         # Less responsive steering
        max_speed=15.0,           # Limit top speed
        continuous=True           # Continuous control is harder
    )
    
    state, info = hard_env.reset(seed=42)
    print(f"Hard Environment Parameters:")
    print(f"  Track Width: {hard_env.custom_track_width} - Narrower track")
    print(f"  Track Length Factor: {hard_env.track_length_factor} - Longer track")
    print(f"  Playfield Size: {hard_env.custom_playfield} - Smaller playfield")
    print(f"  Tile Reward: {hard_env.tile_visited_reward} - Less reward")
    print(f"  Frame Cost: {hard_env.frame_cost} - More penalty")
    print(f"  Engine Power: {hard_env.engine_power} - Weaker engine")
    print(f"  Friction: {hard_env.friction} - More friction")
    print(f"  Max Speed: {hard_env.max_speed} - Speed limited")
    
    # Similar testing loop as above
    total_reward = 0
    steps = 0
    
    try:
        while True:
            action = hard_env.action_space.sample()
            state, reward, terminated, truncated, info = hard_env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"Step {steps}: Total Reward = {total_reward:.2f}")
            
            if terminated or truncated:
                print(f"Hard environment - Episode ended after {steps} steps with reward {total_reward:.2f}")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    hard_env.close()


# Play with keyboard controls
if __name__ == "__main__" and False:  # Set to True to enable
    import pygame
    
    def play_with_keyboard():
        pygame.init()
        env = CustomCarRacing(
            render_mode="human",
            track_width=8.0,  # Wider track for easier play
            engine_power=120.0,  # Stronger engine
            friction=0.9       # Less friction
        )
        
        env.reset()
        total_reward = 0.0
        steps = 0
        quit = False
        action = np.array([0.0, 0.0, 0.0])  # steer, gas, brake
        
        while not quit:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action[0] = -1.0
                    if event.key == pygame.K_RIGHT:
                        action[0] = +1.0
                    if event.key == pygame.K_UP:
                        action[1] = +1.0
                    if event.key == pygame.K_DOWN:
                        action[2] = +0.8
                    if event.key == pygame.K_ESCAPE:
                        quit = True
                        
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT and action[0] < 0:
                        action[0] = 0
                    if event.key == pygame.K_RIGHT and action[0] > 0:
                        action[0] = 0
                    if event.key == pygame.K_UP:
                        action[1] = 0
                    if event.key == pygame.K_DOWN:
                        action[2] = 0
                        
                if event.type == pygame.QUIT:
                    quit = True
                    
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"Step {steps}: Total Reward = {total_reward:.2f}")
                
            if terminated or truncated:
                print(f"Episode ended after {steps} steps with reward {total_reward:.2f}")
                env.reset()
                total_reward = 0.0
                steps = 0
                action = np.array([0.0, 0.0, 0.0])
                
        env.close()
        pygame.quit()
        
    # Uncomment to play with keyboard
    # play_with_keyboard() 