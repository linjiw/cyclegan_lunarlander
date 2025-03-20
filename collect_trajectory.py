import numpy as np
import gymnasium as gym
import time
import os
from custom_lunar_lander import LeftWindLunarLander, RightWindLunarLander

def collect_trajectories(env_id, num_trajectories=1000, trajectory_length=100, render=False):
    """
    Collect trajectories from the environment using random actions.
    Each trajectory is a fixed length sequence of state-action pairs.
    
    Args:
        env_id: The environment ID to collect from
        num_trajectories: Number of trajectories to collect
        trajectory_length: Length of each trajectory (number of steps)
        render: Whether to render the environment
    
    Returns:
        Dictionary with trajectories data
    """
    # Create environment
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    
    # Arrays to store complete trajectories
    all_trajectories = {
        'states': [],     # List of state sequences
        'actions': [],    # List of action sequences
        'rewards': [],    # List of reward sequences
        'next_states': [], # List of next state sequences
        'dones': []       # List of terminal flags
    }
    
    print(f"Collecting {num_trajectories} trajectories of length {trajectory_length} from {env_id}...")
    
    for traj_idx in range(num_trajectories):
        # Reset the environment to start a new trajectory
        state, _ = env.reset(seed=traj_idx)  # Different seed for each trajectory
        
        # Arrays for current trajectory
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Collect one trajectory of specified length
        for step in range(trajectory_length):
            # Select random action
            action = env.action_space.sample()
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            # Move to next state
            state = next_state
            
            # If episode terminated early, we need to reset and continue
            if done and step < trajectory_length - 1:
                state, _ = env.reset(seed=traj_idx * 1000 + step)
        
        # Add this trajectory to our collection
        all_trajectories['states'].append(states)
        all_trajectories['actions'].append(actions)
        all_trajectories['rewards'].append(rewards)
        all_trajectories['next_states'].append(next_states)
        all_trajectories['dones'].append(dones)
        
        # Print progress
        if (traj_idx + 1) % 10 == 0:
            print(f"Collected {traj_idx + 1} trajectories")
    
    env.close()
    
    # Convert list of trajectories to numpy arrays
    for key in all_trajectories:
        all_trajectories[key] = np.array(all_trajectories[key])
    
    print(f"Collection complete: {num_trajectories} trajectories of length {trajectory_length}")
    print(f"Total steps collected: {num_trajectories * trajectory_length}")
    
    return all_trajectories

def save_trajectories(trajectories, filename):
    """
    Save trajectories to a numpy file
    
    Args:
        trajectories: Dictionary of trajectory data
        filename: Filename to save to
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save transitions
    np.save(f"data/{filename}", trajectories)
    
    # Get shapes for reporting
    num_trajectories = trajectories['states'].shape[0]
    trajectory_length = trajectories['states'].shape[1]
    state_dim = trajectories['states'].shape[2]
    
    print(f"Saved trajectories to data/{filename}")
    print(f"Dataset shape: {num_trajectories} trajectories, {trajectory_length} steps each, {state_dim} state dimensions")

def main():
    # Number of trajectories and length parameters
    num_trajectories = 1000
    trajectory_length = 100
    
    # Ensure our custom environments are registered
    if "LunarLanderLeftWind-v0" not in gym.envs.registry.keys():
        gym.register(
            id="LunarLanderLeftWind-v0",
            entry_point=LeftWindLunarLander,
            max_episode_steps=1000,
            reward_threshold=200,
        )
    
    if "LunarLanderRightWind-v0" not in gym.envs.registry.keys():
        gym.register(
            id="LunarLanderRightWind-v0",
            entry_point=RightWindLunarLander,
            max_episode_steps=1000,
            reward_threshold=200,
        )
    
    # Collect trajectories from standard LunarLander for reference
    print("\n=== Collecting from Standard LunarLander ===")
    trajectories_standard = collect_trajectories(
        "LunarLander-v3", 
        num_trajectories=num_trajectories, 
        trajectory_length=trajectory_length,
        render=False
    )
    save_trajectories(trajectories_standard, "lunar_lander_standard_trajectories.npy")
    
    # Wait a moment before starting the next collection
    time.sleep(2)
    
    # Collect trajectories from Left Wind LunarLander
    print("\n=== Collecting from Left Wind LunarLander ===")
    trajectories_left = collect_trajectories(
        "LunarLanderLeftWind-v0", 
        num_trajectories=num_trajectories, 
        trajectory_length=trajectory_length,
        render=False
    )
    save_trajectories(trajectories_left, "lunar_lander_left_wind_trajectories.npy")
    
    # Wait a moment before starting the next collection
    time.sleep(2)
    
    # Collect trajectories from Right Wind LunarLander
    print("\n=== Collecting from Right Wind LunarLander ===")
    trajectories_right = collect_trajectories(
        "LunarLanderRightWind-v0", 
        num_trajectories=num_trajectories, 
        trajectory_length=trajectory_length,
        render=False
    )
    save_trajectories(trajectories_right, "lunar_lander_right_wind_trajectories.npy")
    
    # Print summary of collected data
    print("\n=== Collection Summary ===")
    print(f"Standard LunarLander: {num_trajectories} trajectories, {trajectory_length} steps each")
    print(f"Left Wind LunarLander: {num_trajectories} trajectories, {trajectory_length} steps each")
    print(f"Right Wind LunarLander: {num_trajectories} trajectories, {trajectory_length} steps each")
    print(f"Total steps collected: {num_trajectories * trajectory_length * 3}")
    
    # Calculate average rewards per trajectory
    avg_reward_standard = np.mean([np.sum(rewards) for rewards in trajectories_standard['rewards']])
    avg_reward_left = np.mean([np.sum(rewards) for rewards in trajectories_left['rewards']])
    avg_reward_right = np.mean([np.sum(rewards) for rewards in trajectories_right['rewards']])
    
    # Print statistics for comparison
    print("\n=== Environment Comparison (Average Reward per Trajectory) ===")
    print(f"Standard LunarLander: {avg_reward_standard:.4f}")
    print(f"Left Wind LunarLander: {avg_reward_left:.4f}")
    print(f"Right Wind LunarLander: {avg_reward_right:.4f}")

if __name__ == "__main__":
    main() 