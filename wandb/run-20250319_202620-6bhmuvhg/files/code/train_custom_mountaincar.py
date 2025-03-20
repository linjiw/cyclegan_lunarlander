import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import sys
import time
from datetime import datetime
import wandb  # Import wandb

# Import PPO algorithm
# Fix the import statement - make sure it matches your actual PPO module structure
try:
    from PPO_PyTorch.PPO import PPO
except ImportError:
    # Try alternative import paths
    try:
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
        'force': 0.0008,        # Weaker engine
        'gravity': 0.003,       # More gravity
        'max_speed': 0.06,      # Lower max speed
        'goal_position': 0.55,  # Harder goal
        'init_position_low': -1.1,
        'init_position_high': -0.9
    }
)

def train(env_name, has_continuous_action_space, action_std_init, random_seed):
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    # Create log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("PPO_logs", f"{env_name}_{timestamp}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Logging to {log_dir}")
    
    # Create model directory
    model_dir = os.path.join("PPO_preTrained", env_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f"Saving models to {model_dir}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize wandb
    run = wandb.init(
        project="custom-mountaincar-ppo",
        config={
            "env_name": env_name,
            "has_continuous_action_space": has_continuous_action_space,
            "action_std_init": action_std_init,
            "random_seed": random_seed,
            "lr_actor": 0.0003,
            "lr_critic": 0.001,
            "gamma": 0.99,
            "K_epochs": 40,
            "eps_clip": 0.2,
            "update_timestep": 4000,
        },
        name=f"{env_name}_{timestamp}",
        sync_tensorboard=True,  # Sync TensorBoard logs
        monitor_gym=True,       # Monitor gymnasium environments
        save_code=True,         # Save code for reproducibility
    )
    
    # Training hyperparameters
    max_episodes = 2000                 # Max training episodes
    max_timesteps = 500                 # Max timesteps in one episode
    
    update_timestep = 4000              # Update policy every n timesteps
    K_epochs = 40                       # Update policy for K epochs
    eps_clip = 0.2                      # Clip parameter for PPO
    gamma = 0.99                        # Discount factor
    lr_actor = 0.0003                   # Learning rate for actor
    lr_critic = 0.001                   # Learning rate for critic
    
    # Initialize environment
    print(f"Training environment: {env_name}")
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Set random seeds
    if random_seed:
        print(f"Setting random seed to {random_seed}")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        env.reset(seed=random_seed)
    
    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                   has_continuous_action_space, action_std_init)
    
    # Initialize logging variables
    time_step = 0
    i_episode = 0
    
    # Training loop
    print("Starting training...")
    
    # Keep track of episode rewards and lengths for logging
    episode_rewards = []
    episode_lengths = []
    
    while i_episode < max_episodes:
        state, _ = env.reset()
        current_ep_reward = 0
        current_ep_length = 0
        
        for t in range(max_timesteps):
            time_step += 1
            
            # Select action with policy
            action = ppo_agent.select_action(state)
            
            # Take action in env
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Save in buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(terminated or truncated)
            
            state = next_state
            current_ep_reward += reward
            current_ep_length += 1
            
            # Update if its time
            if time_step % update_timestep == 0:
                print(f"Updating policy at timestep {time_step}...")
                loss_info = ppo_agent.update()
                
                # Log to both TensorBoard and wandb
                writer.add_scalar('Loss/actor', loss_info['actor_loss'], time_step)
                writer.add_scalar('Loss/critic', loss_info['critic_loss'], time_step)
                writer.add_scalar('Loss/entropy', loss_info['entropy'], time_step)
                
                wandb.log({
                    'Loss/actor': loss_info['actor_loss'],
                    'Loss/critic': loss_info['critic_loss'],
                    'Loss/entropy': loss_info['entropy'],
                    'timestep': time_step
                })
            
            # If episode is done, reset environment
            if terminated or truncated:
                break
        
        i_episode += 1
        episode_rewards.append(current_ep_reward)
        episode_lengths.append(current_ep_length)
        
        # Log episode info to both TensorBoard and wandb
        print(f"Episode {i_episode} | Reward: {current_ep_reward:.2f} | Length: {current_ep_length}")
        
        writer.add_scalar('Metrics/episode_reward', current_ep_reward, i_episode)
        writer.add_scalar('Metrics/episode_length', current_ep_length, i_episode)
        
        wandb.log({
            'Metrics/episode_reward': current_ep_reward,
            'Metrics/episode_length': current_ep_length,
            'episode': i_episode
        })
        
        # Calculate and log moving averages
        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            
            writer.add_scalar('Metrics/avg_reward_10', avg_reward, i_episode)
            writer.add_scalar('Metrics/avg_length_10', avg_length, i_episode)
            
            wandb.log({
                'Metrics/avg_reward_10': avg_reward,
                'Metrics/avg_length_10': avg_length,
                'episode': i_episode
            })
            
            print(f"Last 10 episodes: Avg reward: {avg_reward:.2f} | Avg length: {avg_length:.2f}")
        
        # Save model periodically
        if i_episode % 100 == 0:
            print(f"Saving model at episode {i_episode}...")
            model_path = os.path.join(model_dir, f"{env_name}_{i_episode}.pth")
            ppo_agent.save(model_path)
            
            # Log model artifact to wandb
            artifact = wandb.Artifact(f"model-{env_name}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"{env_name}_final.pth")
    ppo_agent.save(final_model_path)
    
    # Log final model artifact to wandb
    final_artifact = wandb.Artifact(f"model-{env_name}-final", type="model")
    final_artifact.add_file(final_model_path)
    wandb.log_artifact(final_artifact)
    
    env.close()
    writer.close()
    wandb.finish()  # Finish the wandb run
    
    print("============================================================================================")
    print(f"Training finished for {env_name}!")

if __name__ == "__main__":
    # Login to wandb (you'll be prompted to log in if you haven't already)
    wandb.login()
    
    # Train on easy environment
    env_name = "CustomMountainCarEasy-v0"
    has_continuous_action_space = False  # MountainCar has discrete actions
    action_std_init = 0.6  # For continuous action space
    
    print("Training on Easy Environment...")
    train(env_name, has_continuous_action_space, action_std_init, random_seed=42)
    
    # Train on hard environment
    env_name = "CustomMountainCarHard-v0"
    
    print("Training on Hard Environment...")
    train(env_name, has_continuous_action_space, action_std_init, random_seed=42) 