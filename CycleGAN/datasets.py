import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class LunarLanderDataset(Dataset):
    def __init__(self, raw_data_path, mode='train', train_split=0.8, unaligned=True):
        """
        Initialize the LunarLander dataset from raw .npy files
        
        Parameters:
            raw_data_path (string): Directory with raw .npy files
            mode (string): 'train' or 'test'
            train_split (float): Percentage of data to use for training
            unaligned (bool): Whether to randomly sample domain B data
        """
        self.unaligned = unaligned
        
        # Load raw data files
        left_wind_file = os.path.join(raw_data_path, 'lunar_lander_left_wind_trajectories_concatenated.npy')
        right_wind_file = os.path.join(raw_data_path, 'lunar_lander_right_wind_trajectories_concatenated.npy')
        
        if not os.path.exists(left_wind_file) or not os.path.exists(right_wind_file):
            raise ValueError(f"Raw data files not found in {raw_data_path}")
        
        # Load the numpy arrays
        data_A = np.load(left_wind_file)  # Domain A (left wind)
        data_B = np.load(right_wind_file)  # Domain B (right wind)
        
        print(f"Raw data shapes - A: {data_A.shape}, B: {data_B.shape}")
        
        # Convert to torch tensors
        data_A = torch.from_numpy(data_A).float()
        data_B = torch.from_numpy(data_B).float()
        
        # Normalize data to [-1, 1] range for better Tanh activation performance
        data_A = self._normalize_data(data_A)
        data_B = self._normalize_data(data_B)
        
        # Split into train and test
        n_samples_A = data_A.shape[0]
        n_samples_B = data_B.shape[0]
        
        # Calculate split indices
        train_idx_A = int(n_samples_A * train_split)
        train_idx_B = int(n_samples_B * train_split)
        
        if mode == 'train':
            self.data_A = data_A[:train_idx_A]
            self.data_B = data_B[:train_idx_B]
        else:  # test
            self.data_A = data_A[train_idx_A:]
            self.data_B = data_B[train_idx_B:]
        
        print(f"Loaded {mode} data: Domain A: {self.data_A.shape}, Domain B: {self.data_B.shape}")
    
    def _normalize_data(self, data):
        """
        Normalize data to range [-1, 1] to work better with Tanh activation
        """
        # Check if data needs normalization
        min_val = torch.min(data)
        max_val = torch.max(data)
        
        if min_val >= -1 and max_val <= 1:
            return data  # Already normalized
            
        # Apply normalization
        data_min = torch.min(data, dim=0)[0]
        data_max = torch.max(data, dim=0)[0]
        
        # Handle constant features (avoid division by zero)
        range_val = data_max - data_min
        range_val[range_val == 0] = 1.0
        
        normalized = 2 * (data - data_min) / range_val - 1
        
        print(f"Normalized data range: [{torch.min(normalized).item():.3f}, {torch.max(normalized).item():.3f}]")
        return normalized
    
    def __getitem__(self, index):
        """
        Return a data pair
        
        Parameters:
            index (int): Index
            
        Returns:
            Dictionary containing A and B data vectors
        """
        item_A = self.data_A[index % len(self.data_A)]
        
        if self.unaligned:
            item_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
        else:
            item_B = self.data_B[index % len(self.data_B)]
        
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        """
        Return the total number of data samples
        """
        return max(len(self.data_A), len(self.data_B))

class VectorDataset(Dataset):
    def __init__(self, root, dimension=1000, mode='train'):
        """
        Initialize the VectorDataset
        
        Parameters:
            root (string): Directory with domain A and B data
            dimension (int): Dimension of the data vectors
            mode (string): 'train' or 'test'
        """
        self.dimension = dimension
        
        # Load data
        self.data_A = self._load_data(os.path.join(root, f'{mode}/A'))
        self.data_B = self._load_data(os.path.join(root, f'{mode}/B'))
        
    def _load_data(self, path):
        """
        Load numpy or torch data files from a directory
        
        Parameters:
            path (string): Path to the directory containing the data files
            
        Returns:
            List of data vectors
        """
        data = []
        # Check if directory exists
        if not os.path.exists(path):
            raise ValueError(f"Data directory {path} does not exist")
            
        # Load all .npy or .pt files
        for file in os.listdir(path):
            if file.endswith('.npy'):
                vector = torch.from_numpy(np.load(os.path.join(path, file))).float()
                data.append(vector)
            elif file.endswith('.pt'):
                vector = torch.load(os.path.join(path, file)).float()
                data.append(vector)
        
        return data
    
    def __getitem__(self, index):
        """
        Return a data pair
        
        Parameters:
            index (int): Index
            
        Returns:
            Dictionary containing A and B data vectors
        """
        item_A = self.data_A[index % len(self.data_A)]
        item_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
        
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        """
        Return the total number of data samples
        """
        return max(len(self.data_A), len(self.data_B))


class DummyVectorDataset(Dataset):
    """
    A dummy dataset generator for testing purposes when real data is not available.
    Creates random vectors for domain A and B with different distributions.
    """
    def __init__(self, size=1000, dimension=1000, mode='train'):
        """
        Initialize the dummy dataset
        
        Parameters:
            size (int): Number of samples to generate
            dimension (int): Dimension of the vectors
            mode (string): 'train' or 'test' (affects random seed)
        """
        self.size = size
        self.dimension = dimension
        
        # Set a different seed for train and test
        seed = 42 if mode == 'train' else 84
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate domain A as random normal with mean 0.5
        self.data_A = [torch.FloatTensor(np.random.normal(0.5, 0.5, dimension)) for _ in range(size)]
        
        # Generate domain B as random normal with mean -0.5
        self.data_B = [torch.FloatTensor(np.random.normal(-0.5, 0.5, dimension)) for _ in range(size)]
        
    def __getitem__(self, index):
        item_A = self.data_A[index % self.size]
        item_B = self.data_B[random.randint(0, self.size - 1)]
        
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return self.size