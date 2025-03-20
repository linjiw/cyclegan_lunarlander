import numpy as np
import gymnasium as gym
import time
import os
from custom_lunar_lander import LeftWindLunarLander, RightWindLunarLander

def collect_transitions(env_id, num_steps=1000, render=False):
    """
    Collect (state, action, next_state, reward, done) transitions from the environment
    using random actions for the specified number of steps.
    
    Args:
        env_id: The environment ID to collect from
        num_steps: Number of steps to collect
        render: Whether to render the environment
    
    Returns:
        Tuple of numpy arrays (states, actions, next_states, rewards, dones)
    """
    # Create environment
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    state, _ = env.reset(seed=42)  # Use fixed seed for reproducibility
    
    # Initialize arrays to store data
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    
    print(f"Collecting {num_steps} transitions from {env_id}...")
    
    step_count = 0
    episode_count = 0
    
    while step_count < num_steps:
        # Select random action
        action = env.action_space.sample()
        
        # Take step in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        
        # Update state
        state = next_state
        step_count += 1
        
        # Print progress
        if step_count % 100 == 0:
            print(f"Collected {step_count} steps")
            
        # Reset if episode ended
        if done:
            episode_count += 1
            state, _ = env.reset()
    
    env.close()
    
    print(f"Collection complete: {step_count} steps across {episode_count} episodes")
    
    # Convert to numpy arrays
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int32),
        np.array(next_states, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.bool_)
    )

def save_transitions(transitions, filename):
    """
    Save transitions to a numpy file
    
    Args:
        transitions: Tuple of (states, actions, next_states, rewards, dones)
        filename: Filename to save to
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save transitions
    np.save(f"data/{filename}", {
        "states": transitions[0],
        "actions": transitions[1],
        "next_states": transitions[2],
        "rewards": transitions[3],
        "dones": transitions[4]
    })
    
    print(f"Saved transitions to data/{filename}")
    print(f"Dataset shape: {transitions[0].shape[0]} transitions, {transitions[0].shape[1]} state dimensions")

def main():
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
    
    # Just collect from our custom environments
    print("\n=== Collecting from Left Wind LunarLander ===")
    transitions_left = collect_transitions("LunarLanderLeftWind-v0", render=True)
    save_transitions(transitions_left, "lunar_lander_left_wind.npy")
    
    # Wait a moment before starting the next collection
    time.sleep(2)
    
    # Collect transitions from Right Wind LunarLander
    print("\n=== Collecting from Right Wind LunarLander ===")
    transitions_right = collect_transitions("LunarLanderRightWind-v0", render=True)
    save_transitions(transitions_right, "lunar_lander_right_wind.npy")
    
    # Optionally, collect from the standard environment with correct version
    print("\n=== Collecting from Standard LunarLander ===")
    transitions_standard = collect_transitions("LunarLander-v3", render=True)
    save_transitions(transitions_standard, "lunar_lander_standard.npy")
    
    # Print summary of collected data
    print("\n=== Collection Summary ===")
    print(f"Left Wind LunarLander: {transitions_left[0].shape[0]} transitions")
    print(f"Right Wind LunarLander: {transitions_right[0].shape[0]} transitions")
    print(f"Standard LunarLander: {transitions_standard[0].shape[0]} transitions")
    
    # Basic statistics to compare environments
    print("\n=== Environment Comparison ===")
    print(f"Left Wind - Avg Reward: {np.mean(transitions_left[3]):.4f}, Std Dev: {np.std(transitions_left[3]):.4f}")
    print(f"Right Wind - Avg Reward: {np.mean(transitions_right[3]):.4f}, Std Dev: {np.std(transitions_right[3]):.4f}")
    print(f"Standard - Avg Reward: {np.mean(transitions_standard[3]):.4f}, Std Dev: {np.std(transitions_standard[3]):.4f}")

if __name__ == "__main__":
    main() 