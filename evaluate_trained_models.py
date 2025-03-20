import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
from datetime import datetime
import time

# Import PPO algorithm
try:
    from PPO_PyTorch.PPO import PPO
except ImportError:
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from PPO import PPO
    except ImportError:
        raise ImportError("Could not import PPO module. Please check the path and module name.")

# Import the custom environment
from custom_mountain_car import CustomMountainCar

# Register the environments
gym.register(
    id="CustomMountainCarEasy-v0",
    entry_point=CustomMountainCar,
    max_episode_steps=1000,
    kwargs={
        'force': 0.003,         # Stronger engine
        'gravity': 0.0015,      # Less gravity
        'max_speed': 0.1,       # Higher max speed
        'goal_position': 0.45,  # Easier goal
    }
)

gym.register(
    id="CustomMountainCarHard-v0",
    entry_point=CustomMountainCar,
    max_episode_steps=1000,
    kwargs={
        'force': 0.0015,        # Moderately weaker engine
        'gravity': 0.002,       # Moderate gravity
        'max_speed': 0.08,      # Moderate max speed
        'goal_position': 0.5,   # Standard goal position
        'init_position_low': -0.8,  # Less extreme starting position
        'init_position_high': -0.6  # Less extreme starting position
    }
)

def load_model(env_name, state_dim, action_dim, has_continuous_action_space=False):
    """
    Load a pre-trained model
    """
    # Set paths to models
    model_dir = os.path.join("PPO_preTrained", env_name)
    
    # Find the final model or latest model
    final_model_path = os.path.join(model_dir, f"{env_name}_final.pth")
    
    if not os.path.exists(final_model_path):
        # Look for the model with highest episode number
        model_files = [f for f in os.listdir(model_dir) if f.startswith(env_name) and f.endswith(".pth")]
        if not model_files:
            raise FileNotFoundError(f"No trained model found for {env_name}")
        
        # Sort by episode number
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
        final_model_path = os.path.join(model_dir, model_files[-1])
    
    print(f"Loading model: {final_model_path}")
    
    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, 0.0003, 0.001, 0.99, 40, 0.2, 
                   has_continuous_action_space, 0.6)
    
    # Load trained weights
    ppo_agent.load(final_model_path)
    
    return ppo_agent, final_model_path.split('/')[-1]

def evaluate_model(model, env_name, num_episodes=50, render=False, record_trajectory=False):
    """
    Evaluate a model in a given environment
    """
    # Create environment
    env = gym.make(env_name, render_mode="human" if render else None)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    trajectories = [] if record_trajectory else None
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep)  # Use episode number as seed for reproducibility
        
        ep_reward = 0
        ep_length = 0
        done = False
        
        # For recording trajectory
        positions = [] if record_trajectory else None
        velocities = [] if record_trajectory else None
        
        while not done:
            # Select action
            action = model.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Record state if tracking trajectory
            if record_trajectory:
                positions.append(next_state[0])
                velocities.append(next_state[1])
            
            # Update metrics
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated
            state = next_state
            
            # Safety check for max steps - shouldn't be needed with gymnasium, but just in case
            if ep_length >= 1000:
                break
        
        # Record if agent reached goal
        if terminated:  # Only terminated, not truncated - means agent reached goal
            success_count += 1
        
        # Store episode metrics
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        # Store trajectory data if recording
        if record_trajectory:
            trajectories.append({
                'positions': positions,
                'velocities': velocities,
                'reward': ep_reward,
                'length': ep_length,
                'success': terminated
            })
        
        # Print progress
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes} - Avg. Reward: {np.mean(episode_rewards[-10:]):.2f}")
    
    # Calculate final metrics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    env.close()
    
    results = {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths,
        'trajectories': trajectories
    }
    
    return results

def create_comparison_table(results_dict):
    """
    Create a comparison table of results
    """
    # Prepare data for the table
    data = {
        'Model → Environment': [],
        'Avg. Reward': [],
        'Avg. Episode Length': [],
        'Success Rate (%)': []
    }
    
    for key, result in results_dict.items():
        data['Model → Environment'].append(key)
        data['Avg. Reward'].append(f"{result['avg_reward']:.2f}")
        data['Avg. Episode Length'].append(f"{result['avg_length']:.2f}")
        data['Success Rate (%)'].append(f"{result['success_rate'] * 100:.1f}%")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def plot_rewards_distribution(results_dict, filename="reward_distributions.png"):
    """
    Plot reward distributions for each combination
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (key, result) in enumerate(results_dict.items()):
        ax = axes[i]
        rewards = result['all_rewards']
        
        # Create histogram
        ax.hist(rewards, bins=20, alpha=0.7, color=f'C{i}')
        ax.axvline(result['avg_reward'], color='red', linestyle='dashed', linewidth=1, 
                  label=f'Mean: {result["avg_reward"]:.2f}')
        
        ax.set_title(f"{key}")
        ax.set_xlabel("Episode Reward")
        ax.set_ylabel("Frequency")
        ax.legend()
        
        # Add success rate
        success_text = f"Success Rate: {result['success_rate'] * 100:.1f}%"
        ax.annotate(success_text, xy=(0.05, 0.95), xycoords='axes fraction',
                   verticalalignment='top', fontsize=10, 
                   bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return filename

def plot_trajectories(results_dict, filename="trajectories.png"):
    """
    Plot sample trajectories for each combination
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (key, result) in enumerate(results_dict.items()):
        ax = axes[i]
        
        # Only plot if we have trajectory data
        if result['trajectories'] is not None:
            # Plot a few successful trajectories
            successful_trajectories = [t for t in result['trajectories'] if t['success']]
            # If no successful trajectories, just plot the first few
            trajectories_to_plot = successful_trajectories[:3] if successful_trajectories else result['trajectories'][:3]
            
            for j, traj in enumerate(trajectories_to_plot[:5]):  # Plot at most 5 trajectories
                ax.plot(traj['positions'], traj['velocities'], 
                       alpha=0.7, label=f"Episode {j+1}: {traj['reward']:.1f}")
            
            ax.set_title(f"{key}")
            ax.set_xlabel("Position")
            ax.set_ylabel("Velocity")
            ax.legend()
            
            # Mark goal position
            goal_position = 0.45 if "Easy" in key.split(' → ')[1] else 0.5
            ax.axvline(goal_position, color='green', linestyle='dotted')
        else:
            ax.text(0.5, 0.5, "No trajectory data", 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return filename

def main():
    # Parameters
    num_eval_episodes = 50  # Number of episodes to evaluate each combination
    render_final_episode = True  # Render the final episode of each evaluation
    record_trajectories = True  # Record trajectories for visualization
    
    # Create results directory
    results_dir = f"evaluation_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results to: {results_dir}")
    
    # Environment parameters
    has_continuous_action_space = False
    state_dim = 2
    action_dim = 3
    
    # Load models
    easy_model, easy_model_name = load_model("CustomMountainCarEasy-v0", state_dim, action_dim, has_continuous_action_space)
    hard_model, hard_model_name = load_model("CustomMountainCarHard-v0", state_dim, action_dim, has_continuous_action_space)
    
    # Evaluate each model in each environment
    print("\n1. Evaluating Easy model in Easy environment...")
    easy_in_easy = evaluate_model(
        easy_model, "CustomMountainCarEasy-v0", 
        num_episodes=num_eval_episodes, 
        render=render_final_episode,
        record_trajectory=record_trajectories
    )
    
    print("\n2. Evaluating Easy model in Hard environment...")
    easy_in_hard = evaluate_model(
        easy_model, "CustomMountainCarHard-v0", 
        num_episodes=num_eval_episodes, 
        render=render_final_episode,
        record_trajectory=record_trajectories
    )
    
    print("\n3. Evaluating Hard model in Easy environment...")
    hard_in_easy = evaluate_model(
        hard_model, "CustomMountainCarEasy-v0", 
        num_episodes=num_eval_episodes, 
        render=render_final_episode,
        record_trajectory=record_trajectories
    )
    
    print("\n4. Evaluating Hard model in Hard environment...")
    hard_in_hard = evaluate_model(
        hard_model, "CustomMountainCarHard-v0", 
        num_episodes=num_eval_episodes, 
        render=render_final_episode,
        record_trajectory=record_trajectories
    )
    
    # Collect results
    results = {
        "Easy Model → Easy Env": easy_in_easy,
        "Easy Model → Hard Env": easy_in_hard,
        "Hard Model → Easy Env": hard_in_easy,
        "Hard Model → Hard Env": hard_in_hard
    }
    
    # Create comparison table
    comparison_table = create_comparison_table(results)
    print("\nResults:")
    print(comparison_table)
    
    # Save table to CSV
    table_path = os.path.join(results_dir, "comparison_results.csv")
    comparison_table.to_csv(table_path, index=False)
    print(f"Saved comparison table to: {table_path}")
    
    # Plot reward distributions
    reward_plot_path = os.path.join(results_dir, "reward_distributions.png")
    plot_rewards_distribution(results, reward_plot_path)
    print(f"Saved reward distributions to: {reward_plot_path}")
    
    # Plot sample trajectories
    if record_trajectories:
        trajectory_plot_path = os.path.join(results_dir, "sample_trajectories.png")
        plot_trajectories(results, trajectory_plot_path)
        print(f"Saved sample trajectories to: {trajectory_plot_path}")
    
    # Save detailed results to file
    detailed_results = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "easy_model": easy_model_name,
        "hard_model": hard_model_name,
        "num_eval_episodes": num_eval_episodes,
        "results": {
            k: {
                "avg_reward": v["avg_reward"],
                "avg_length": v["avg_length"],
                "success_rate": v["success_rate"],
                "all_rewards": v["all_rewards"],
                "all_lengths": v["all_lengths"]
            } for k, v in results.items()
        }
    }
    
    # Save as CSV files for each combination
    for key, result in results.items():
        model_env_name = key.replace(" → ", "_to_").replace(" ", "_")
        df = pd.DataFrame({
            'episode': range(1, num_eval_episodes + 1),
            'reward': result['all_rewards'],
            'length': result['all_lengths'],
        })
        df_path = os.path.join(results_dir, f"{model_env_name}_episodes.csv")
        df.to_csv(df_path, index=False)
    
    print("\nEvaluation complete! Results saved to:", results_dir)
    print("\nSummary of findings:")
    
    # Add some analysis text based on the results
    best_combo = comparison_table.iloc[comparison_table['Success Rate (%)'].str.rstrip('%').astype(float).idxmax()]
    print(f"Best performing combination: {best_combo['Model → Environment']} with success rate {best_combo['Success Rate (%)']}")
    
    # Evaluate transfer learning effectiveness
    try:
        easy_success = float(comparison_table.loc[comparison_table['Model → Environment'] == 'Easy Model → Easy Env', 'Success Rate (%)'].values[0].rstrip('%'))
        easy_transfer = float(comparison_table.loc[comparison_table['Model → Environment'] == 'Easy Model → Hard Env', 'Success Rate (%)'].values[0].rstrip('%'))
        
        hard_success = float(comparison_table.loc[comparison_table['Model → Environment'] == 'Hard Model → Hard Env', 'Success Rate (%)'].values[0].rstrip('%'))
        hard_transfer = float(comparison_table.loc[comparison_table['Model → Environment'] == 'Hard Model → Easy Env', 'Success Rate (%)'].values[0].rstrip('%'))
        
        print(f"\nTransfer learning effectiveness:")
        print(f"Easy Model: {easy_transfer/easy_success*100:.1f}% of original performance when transferred to hard environment")
        print(f"Hard Model: {hard_transfer/hard_success*100:.1f}% of original performance when transferred to easy environment")
        
        if easy_transfer/easy_success > hard_transfer/hard_success:
            print("The Easy model generalizes better to the hard environment than vice versa.")
        else:
            print("The Hard model generalizes better to the easy environment than vice versa.")
    except:
        print("Could not compute transfer learning analysis due to missing data.")

if __name__ == "__main__":
    main() 