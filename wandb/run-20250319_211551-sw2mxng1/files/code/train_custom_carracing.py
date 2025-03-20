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
try:
    from PPO_PyTorch.PPO import PPO, device  # Also import the global device variable
except ImportError:
    # Try alternative import paths
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from PPO import PPO, device
    except ImportError:
        raise ImportError("Could not import PPO module. Please check the path and module name.")

# Import the custom environment
from custom_car_racing import CustomCarRacing

# Register the environments
gym.register(
    id="CustomCarRacingEasy-v0",
    entry_point=CustomCarRacing,
    max_episode_steps=2000,
    kwargs={
        'track_width': 10.0,         # Wider track (original is ~6.67)
        'track_length_factor': 0.8,  # Shorter track
        'engine_power': 130.0,       # Stronger engine (130% of normal)
        'friction': 0.9,             # Less friction (easier to drive)
        'max_steering': 1.8,         # More responsive steering
        'continuous': False,         # Discrete actions for easier control
        'frame_cost': 0.05,          # Less penalty per frame
        'tile_visited_reward': 1200  # More reward
    }
)

gym.register(
    id="CustomCarRacingHard-v0",
    entry_point=CustomCarRacing,
    max_episode_steps=3000,
    kwargs={
        'track_width': 4.0,           # Narrower track
        'track_length_factor': 1.3,   # Longer track
        'playfield_size': 350,        # Smaller playfield (easier to go out of bounds)
        'engine_power': 80.0,         # Weaker engine (80% of normal)
        'friction': 1.2,              # More friction (harder to control)
        'max_steering': 1.2,          # Less responsive steering
        'max_speed': 15.0,            # Limit top speed
        'continuous': True,           # Continuous control is harder
        'frame_cost': 0.15,           # More penalty per frame
        'out_of_bounds_penalty': 150  # Bigger penalty for going off-track
    }
)

# Set up MPS device for Apple Silicon
def setup_mps_device():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
        return mps_device
    elif torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return cuda_device
    else:
        cpu_device = torch.device("cpu")
        print("Using CPU device")
        return cpu_device

# CNN encoder for processing image observations
class CNNEncoder(torch.nn.Module):
    def __init__(self, output_dim):
        super(CNNEncoder, self).__init__()
        # Input: 96x96x3 image
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4)  # -> 23x23x32
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)  # -> 10x10x64
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)  # -> 8x8x64
        
        # Calculate flattened size
        self.fc_size = 8 * 8 * 64
        
        self.fc = torch.nn.Linear(self.fc_size, output_dim)
        
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # Normalize pixel values to [0, 1]
        x = x / 255.0
        
        # Transpose from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(-1, self.fc_size)
        
        x = self.fc(x)
        return x

# Modified PPO for image-based observations
class ImagePPO(PPO):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                has_continuous_action_space, action_std_init=0.6, encoder_dim=512):
        # Initialize parent PPO class
        super(ImagePPO, self).__init__(encoder_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, 
                                     eps_clip, has_continuous_action_space, action_std_init)
        
        # Create CNN encoder
        self.encoder = CNNEncoder(encoder_dim)
        
        # Move encoder to device (using the global device from PPO)
        self.encoder.to(device)
        
        # Override the optimizer to include encoder params
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.encoder.parameters(), 'lr': lr_actor}
        ])
    
    def select_action(self, state):
        # Preprocess state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Encode the image state
        encoded_state = self.encoder(state_tensor)
        
        # Use encoded state for action selection
        with torch.no_grad():
            if self.has_continuous_action_space:
                state_val, action, action_logprob = self.policy_old.act(encoded_state)
                
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
                
                return action.detach().cpu().numpy().flatten()
            else:
                state_val, action, action_logprob = self.policy_old.act(encoded_state)
                
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
                
                return action.item()
    
    def update(self):
        # Process all the states in the buffer through the encoder
        encoded_states = []
        for state in self.buffer.states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            encoded_state = self.encoder(state_tensor).detach()
            encoded_states.append(encoded_state)
        
        # Replace states in buffer with encoded states
        temp_states = self.buffer.states.copy()
        self.buffer.states = encoded_states
        
        # Call parent update method
        update_info = super().update()
        
        # Restore original states
        self.buffer.states = temp_states
        
        return update_info
    
    def save(self, filepath):
        """Save the model with the encoder"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'has_continuous_action_space': self.has_continuous_action_space,
            'action_std': self.action_std if hasattr(self, 'action_std') else None
        }, filepath)
        
    def load(self, filepath):
        """Load the model with the encoder"""
        checkpoint = torch.load(filepath, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['action_std'] is not None:
            self.action_std = checkpoint['action_std']
            self.set_action_std(self.action_std)
        print("Model loaded successfully")


def train(env_name, has_continuous_action_space, action_std_init, random_seed):
    print("============================================================================================")
    
    # Set the device to MPS (Apple Silicon)
    custom_device = setup_mps_device()
    
    # Modify the global device in the PPO module
    global device
    device = custom_device
    
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
        project="custom-carracing-ppo",
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
            "encoder_dim": 512,
            "device": device.type
        },
        name=f"{env_name}_{timestamp}",
        sync_tensorboard=True,  # Sync TensorBoard logs
        monitor_gym=True,       # Monitor gymnasium environments
        save_code=True,         # Save code for reproducibility
    )
    
    # Training hyperparameters
    max_episodes = 5000                # Max training episodes (more for image-based tasks)
    max_timesteps = 3000               # Max timesteps in one episode (Increased for Car Racing)
    
    update_timestep = 4000             # Update policy every n timesteps
    K_epochs = 40                      # Update policy for K epochs
    eps_clip = 0.2                     # Clip parameter for PPO
    gamma = 0.99                       # Discount factor
    lr_actor = 0.0003                  # Learning rate for actor
    lr_critic = 0.001                  # Learning rate for critic
    encoder_dim = 512                  # Dimension of encoded state representation
    
    # Initialize environment
    print(f"Training environment: {env_name}")
    env = gym.make(env_name)
    
    # For image observations, we pass the raw image dimensions
    state_dim = env.observation_space.shape
    
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
    
    # Initialize modified PPO agent with CNN encoder
    ppo_agent = ImagePPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                       has_continuous_action_space, action_std_init, encoder_dim)
    
    # Initialize logging variables
    time_step = 0
    i_episode = 0
    
    # Training loop
    print("Starting training...")
    
    # Keep track of episode rewards and lengths for logging
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    
    while i_episode < max_episodes:
        state, _ = env.reset()
        current_ep_reward = 0
        current_ep_length = 0
        
        for t in range(max_timesteps):
            time_step += 1
            
            # Select action with policy
            action = ppo_agent.select_action(state)
            
            # Take action in env
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Save in buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(terminated or truncated)
            
            state = next_state
            current_ep_reward += reward
            current_ep_length += 1
            
            # Update if its time
            if time_step % update_timestep == 0:
                print(f"Updating policy at timestep {time_step}...")
                update_info = ppo_agent.update()
                
                # Log that an update occurred
                writer.add_scalar('Training/policy_update', 1, time_step)
                
                # Log loss metrics if available
                if update_info:
                    for k, v in update_info.items():
                        writer.add_scalar(f'Training/{k}', v, time_step)
                        wandb.log({f'Training/{k}': v, 'timestep': time_step})
                
                wandb.log({
                    'Training/policy_update': 1,
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
        
        # Log additional info that might be relevant to Car Racing
        if 'lap_finished' in info:
            writer.add_scalar('Metrics/lap_finished', 1 if info['lap_finished'] else 0, i_episode)
            wandb.log({'Metrics/lap_finished': 1 if info['lap_finished'] else 0})
        
        if 'tile_visited_count' in info:
            writer.add_scalar('Metrics/tiles_visited', info['tile_visited_count'], i_episode)
            wandb.log({'Metrics/tiles_visited': info['tile_visited_count']})
        
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
        
        # Save the best model based on reward
        if current_ep_reward > best_reward:
            best_reward = current_ep_reward
            best_model_path = os.path.join(model_dir, f"{env_name}_best.pth")
            ppo_agent.save(best_model_path)
            print(f"New best model saved with reward {best_reward:.2f}")
    
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
    # Login to wandb
    wandb.login()
    
    # Train on easy environment
    env_name = "CustomCarRacingEasy-v0"
    has_continuous_action_space = False  # Using discrete actions for easier training
    action_std_init = 0.6  # For continuous action space
    
    print("Training on Easy Environment...")
    train(env_name, has_continuous_action_space, action_std_init, random_seed=42)
    
    # Wait a bit before starting the next training
    time.sleep(5)
    
    # Train on hard environment
    env_name = "CustomCarRacingHard-v0"
    has_continuous_action_space = True  # Using continuous actions for hard version
    
    print("Training on Hard Environment...")
    train(env_name, has_continuous_action_space, action_std_init, random_seed=42) 