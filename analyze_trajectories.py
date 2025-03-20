import numpy as np
import os

def analyze_trajectory_file(file_path):
    """
    Analyze the shape and data types of a trajectory data file.
    
    Args:
        file_path: Path to the .npy file containing trajectory data
    """
    print(f"\n=== Analyzing Trajectory File: {file_path} ===")
    
    # Load data
    try:
        data = np.load(file_path, allow_pickle=True).item()
        
        # Overall structure
        print("Keys in the dataset:", list(data.keys()))
        
        # Analyze states
        states = data['states']
        print("\nStates:")
        print(f"  Shape: {states.shape}")
        print(f"  Type: {states.dtype}")
        print(f"  Min value: {np.min(states)}")
        print(f"  Max value: {np.max(states)}")
        print(f"  Example (first state in first trajectory): {states[0, 0]}")
        
        # Analyze actions
        actions = data['actions']
        print("\nActions:")
        print(f"  Shape: {actions.shape}")
        print(f"  Type: {actions.dtype}")
        if actions.dtype == np.int32 or actions.dtype == np.int64:
            print(f"  Action distribution: {np.bincount(actions.flatten())}")
            print(f"  Unique actions: {np.unique(actions)}")
        else:
            print(f"  Min value: {np.min(actions)}")
            print(f"  Max value: {np.max(actions)}")
        print(f"  Example (first action in first trajectory): {actions[0, 0]}")
        
        # Memory usage (approximate)
        states_size = states.nbytes / (1024 * 1024)  # MB
        actions_size = actions.nbytes / (1024 * 1024)  # MB
        
        print("\nMemory Usage:")
        print(f"  States: {states_size:.2f} MB")
        print(f"  Actions: {actions_size:.2f} MB")
        
        # Return key information for state-action extraction
        return {
            'num_trajectories': states.shape[0],
            'trajectory_length': states.shape[1],
            'state_dim': states.shape[2] if len(states.shape) > 2 else 1,
            'action_dim': actions.shape[2] if len(actions.shape) > 2 else 1,
            'state_type': states.dtype,
            'action_type': actions.dtype
        }
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def extract_state_action_data(file_path, output_file=None):
    """
    Extract only the state and action data from a trajectory file and save to a new file.
    
    Args:
        file_path: Path to the original trajectory file
        output_file: Path to save the extracted data (if None, will generate a name)
    
    Returns:
        Path to the output file
    """
    print(f"\n=== Extracting State-Action Data from: {file_path} ===")
    
    # Load data
    data = np.load(file_path, allow_pickle=True).item()
    
    # Extract only states and actions
    extracted_data = {
        'states': data['states'],
        'actions': data['actions']
    }
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = f"data/{name_without_ext}_state_action.npy"
    
    # Save the extracted data
    np.save(output_file, extracted_data)
    
    # Get sizes
    states_size = extracted_data['states'].nbytes / (1024 * 1024)  # MB
    actions_size = extracted_data['actions'].nbytes / (1024 * 1024)  # MB
    total_size = (states_size + actions_size)
    
    print(f"Extracted data saved to: {output_file}")
    print(f"Total size of extracted data: {total_size:.2f} MB")
    
    return output_file

def create_concatenated_trajectories(file_path, output_file=None):
    """
    Create sequences where each state is followed by its action in a flattened array.
    Format: [state_1, action_1, state_2, action_2, ...]
    
    Args:
        file_path: Path to the trajectory file
        output_file: Path to save the concatenated data (if None, will generate a name)
    
    Returns:
        Path to the output file
    """
    print(f"\n=== Creating Concatenated State-Action Sequences from: {file_path} ===")
    
    # Load data
    data = np.load(file_path, allow_pickle=True).item()
    states = data['states']
    actions = data['actions']
    
    num_trajectories = states.shape[0]
    trajectory_length = states.shape[1]
    state_dim = states.shape[2] if len(states.shape) > 2 else 1
    
    # Determine if actions are discrete or continuous
    if actions.dtype == np.int32 or actions.dtype == np.int64:
        # For discrete actions, we need to convert to float and reshape
        actions_reshaped = actions.reshape(num_trajectories, trajectory_length, 1).astype(np.float32)
    else:
        # For continuous actions, just use as is or reshape if needed
        if len(actions.shape) == 3:
            actions_reshaped = actions
        else:
            actions_reshaped = actions.reshape(num_trajectories, trajectory_length, 1)
    
    # Create the concatenated sequences
    concatenated_trajectories = []
    
    for traj_idx in range(num_trajectories):
        sequence = []
        for step_idx in range(trajectory_length):
            # Add state
            state = states[traj_idx, step_idx]
            sequence.extend(state)
            
            # Add action
            action = actions_reshaped[traj_idx, step_idx]
            if len(action.shape) > 0:  # If action is multi-dimensional
                sequence.extend(action)
            else:
                sequence.append(action)
        
        concatenated_trajectories.append(sequence)
    
    # Convert to numpy array
    concatenated_trajectories = np.array(concatenated_trajectories, dtype=np.float32)
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = f"data/{name_without_ext}_concatenated.npy"
    
    # Save the concatenated data
    np.save(output_file, concatenated_trajectories)
    
    # Calculate size
    size_mb = concatenated_trajectories.nbytes / (1024 * 1024)
    
    print(f"Concatenated data saved to: {output_file}")
    print(f"Shape of concatenated data: {concatenated_trajectories.shape}")
    print(f"Size of concatenated data: {size_mb:.2f} MB")
    
    # Calculate expected length of each sequence
    expected_length = trajectory_length * (state_dim + 1)  # state_dim + 1 action dimension
    print(f"Each sequence contains {trajectory_length} steps with {state_dim} state dimensions and 1 action dimension")
    print(f"Expected sequence length: {expected_length} elements")
    
    return output_file

def main():
    # Directory containing trajectory data
    data_dir = "data"
    
    # Trajectory files
    trajectory_files = [
        f"{data_dir}/lunar_lander_standard_trajectories.npy",
        f"{data_dir}/lunar_lander_left_wind_trajectories.npy",
        f"{data_dir}/lunar_lander_right_wind_trajectories.npy"
    ]
    
    file_info = []
    
    # Analyze each file
    for file_path in trajectory_files:
        if os.path.exists(file_path):
            info = analyze_trajectory_file(file_path)
            if info:
                file_info.append(info)
                
                # Extract state-action data
                extract_state_action_data(file_path)
                
                # Create concatenated trajectories
                create_concatenated_trajectories(file_path)
        else:
            print(f"File not found: {file_path}")
    
    # Print summary of all files
    if file_info:
        print("\n=== Summary of All Trajectory Files ===")
        for i, info in enumerate(file_info):
            print(f"\nFile {i+1}:")
            print(f"  Number of trajectories: {info['num_trajectories']}")
            print(f"  Trajectory length: {info['trajectory_length']}")
            print(f"  State dimension: {info['state_dim']}")
            print(f"  Action dimension: {info['action_dim']}")
            print(f"  State type: {info['state_type']}")
            print(f"  Action type: {info['action_type']}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 