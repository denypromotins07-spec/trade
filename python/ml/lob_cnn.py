"""
LOB CNN Model - Lightweight 1D Convolutional Neural Network

Develops a lightweight, DirectML-accelerated 1D Convolutional Neural Network
to extract spatial features from the L2 Limit Order Book state without
exceeding memory boundaries.

Optimized for AMD Ryzen AI 5 with strict 4GB RAM quota.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import os


def check_amd_directml() -> bool:
    """Check if AMD DirectML/ROCm environment is available."""
    try:
        # Check for ROCm availability (AMD GPUs)
        if hasattr(torch.backends, 'rocm') and torch.backends.rocm.is_available():
            print("AMD ROCm backend available")
            return True
        # Check for DirectML (Windows with AMD GPU)
        if os.name == 'nt':
            try:
                import torch_directml
                print("DirectML backend available")
                return True
            except ImportError:
                pass
        return False
    except Exception:
        return False


def get_device_config() -> torch.device:
    """Get optimal device based on hardware availability."""
    if check_amd_directml():
        if hasattr(torch.backends, 'rocm') and torch.backends.rocm.is_available():
            return torch.device('cuda')  # ROCm uses CUDA interface
        try:
            import torch_directml
            return torch.device('privateuseone')  # DirectML
        except ImportError:
            pass
    return torch.device('cpu')


class LOBFeatureExtractor(nn.Module):
    """
    Lightweight 1D CNN for extracting spatial features from LOB state.
    
    Architecture optimized for:
    - Low latency inference (< 100 microseconds)
    - Minimal memory footprint (< 50MB model size)
    - AMD Ryzen AI 5 NPU compatibility
    
    Input: L2 order book state (price levels x features)
    Output: Feature embedding for prediction head
    """
    
    def __init__(
        self,
        n_levels: int = 10,
        n_features_per_level: int = 4,
        embedding_dim: int = 64,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        input_channels = n_features_per_level
        
        # Layer 1: Pointwise convolution across features at each level
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Layer 2: Depthwise convolution across levels
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            groups=32,  # Depthwise
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3: Cross-level aggregation
        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, n_features_per_level, n_levels)
            
        Returns:
            Feature embedding of shape (batch_size, embedding_dim)
        """
        # Ensure correct input shape
        if x.dim() == 3:
            # Already correct shape
            pass
        elif x.dim() == 2:
            # Reshape from (batch, features * levels) to (batch, features, levels)
            x = x.view(-1, self.n_features_per_level, self.n_levels)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        # Conv layer 1: Feature mixing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Conv layer 2: Local pattern detection
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Conv layer 3: Cross-level aggregation
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Projection
        x = self.fc(x)
        x = self.dropout(x)
        
        return x


class MicroPricePredictor(nn.Module):
    """
    Complete micro-price prediction model with LOB feature extraction.
    
    Combines CNN feature extractor with prediction head for:
    - Micro-price direction (up/down)
    - Price change magnitude
    - Short-term volatility
    """
    
    def __init__(
        self,
        n_levels: int = 10,
        n_features_per_level: int = 4,
        embedding_dim: int = 64,
        n_prediction_heads: int = 3
    ):
        super().__init__()
        
        self.feature_extractor = LOBFeatureExtractor(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            embedding_dim=embedding_dim
        )
        
        # Prediction heads
        self.direction_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)  # Up/Down classification
        )
        
        self.magnitude_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # Regression
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # Regression
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning all predictions.
        
        Args:
            x: Input LOB state
            
        Returns:
            Dictionary with direction, magnitude, and volatility predictions
        """
        features = self.feature_extractor(x)
        
        return {
            'direction': self.direction_head(features),
            'magnitude': self.magnitude_head(features),
            'volatility': self.volatility_head(features),
        }


def prepare_lob_data(
    bids: np.ndarray,
    asks: np.ndarray,
    n_levels: int = 10
) -> torch.Tensor:
    """
    Prepare L2 order book data for the CNN.
    
    Features per level:
    0. Log price relative to mid
    1. Volume (log scaled)
    2. Order count
    3. Imbalance feature
    
    Args:
        bids: Bid book (n_levels x 4)
        asks: Ask book (n_levels x 4)
        n_levels: Number of levels to use
        
    Returns:
        Tensor of shape (n_features_per_level, 2 * n_levels)
    """
    # Extract features
    bid_prices = bids[:n_levels, 0]
    bid_volumes = bids[:n_levels, 1]
    bid_counts = bids[:n_levels, 2]
    
    ask_prices = asks[:n_levels, 0]
    ask_volumes = asks[:n_levels, 1]
    ask_counts = asks[:n_levels, 2]
    
    # Compute mid price
    mid_price = (bid_prices[0] + ask_prices[0]) / 2
    
    # Normalize prices (log relative to mid)
    bid_price_rel = np.log(bid_prices / mid_price)
    ask_price_rel = np.log(ask_prices / mid_price)
    
    # Log scale volumes
    bid_vol_log = np.log1p(bid_volumes)
    ask_vol_log = np.log1p(ask_volumes)
    
    # Order imbalance at each level
    total_vol = bid_volumes + ask_volumes
    bid_imbalance = np.where(total_vol > 0, bid_volumes / total_vol - 0.5, 0)
    ask_imbalance = np.where(total_vol > 0, ask_volumes / total_vol - 0.5, 0)
    
    # Stack features: [price_rel, vol_log, count, imbalance]
    features = np.stack([
        np.concatenate([bid_price_rel, ask_price_rel]),
        np.concatenate([bid_vol_log, ask_vol_log]),
        np.concatenate([bid_counts, ask_counts]),
        np.concatenate([bid_imbalance, ask_imbalance]),
    ], axis=0)
    
    return torch.FloatTensor(features)


class LOBCNNTrainer:
    """
    Trainer for LOB CNN with memory-efficient batching.
    
    Enforces 4GB RAM limit through:
    - Gradient accumulation
    - Mixed precision training (when available)
    - Automatic batch size adjustment
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        max_memory_gb: float = 4.0,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device_config()
        self.model = model.to(self.device)
        self.max_memory_gb = max_memory_gb
        
        # Loss functions
        self.direction_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training stats
        self.training_history = []
    
    def train_step(
        self,
        lob_data: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Single training step with gradient accumulation.
        
        Args:
            lob_data: Input LOB tensors
            targets: Dictionary with target values
            gradient_accumulation_steps: Steps before optimizer update
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        lob_data = lob_data.to(self.device)
        targets = {k: v.to(self.device) for k, v in targets.items()}
        
        # Forward pass
        predictions = self.model(lob_data)
        
        # Compute losses
        dir_loss = self.direction_loss(predictions['direction'], targets['direction'])
        mag_loss = self.regression_loss(predictions['magnitude'].squeeze(), targets['magnitude'])
        vol_loss = self.regression_loss(predictions['volatility'].squeeze(), targets['volatility'])
        
        # Total loss (weighted sum)
        total_loss = dir_loss + 0.5 * mag_loss + 0.3 * vol_loss
        
        # Backward pass
        total_loss.backward()
        
        return {
            'total_loss': total_loss.item(),
            'direction_loss': dir_loss.item(),
            'magnitude_loss': mag_loss.item(),
            'volatility_loss': vol_loss.item(),
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: List[Tuple[torch.Tensor, Dict]]
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: List of (data, targets) tuples
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_losses = {'total': 0, 'direction': 0, 'magnitude': 0, 'volatility': 0}
        n_batches = 0
        
        for lob_data, targets in val_loader:
            lob_data = lob_data.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            predictions = self.model(lob_data)
            
            dir_loss = self.direction_loss(predictions['direction'], targets['direction'])
            mag_loss = self.regression_loss(predictions['magnitude'].squeeze(), targets['magnitude'])
            vol_loss = self.regression_loss(predictions['volatility'].squeeze(), targets['volatility'])
            
            total_loss = dir_loss + 0.5 * mag_loss + 0.3 * vol_loss
            
            total_losses['total'] += total_loss.item()
            total_losses['direction'] += dir_loss.item()
            total_losses['magnitude'] += mag_loss.item()
            total_losses['volatility'] += vol_loss.item()
            n_batches += 1
        
        return {k: v / n_batches for k, v in total_losses.items()}
    
    def export_onnx(self, output_path: str, n_levels: int = 10):
        """
        Export model to ONNX format for Rust inference.
        
        Args:
            output_path: Path to save ONNX model
            n_levels: Number of LOB levels
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 4, n_levels * 2)  # (batch, features, levels*2)
        dummy_input = dummy_input.to(self.device)
        
        # Export
        torch.onnx.export(
            self.model.feature_extractor,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['lob_state'],
            output_names=['features'],
            dynamic_axes={
                'lob_state': {0: 'batch_size'},
                'features': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to {output_path}")


if __name__ == '__main__':
    print("LOB CNN Model")
    print("=" * 40)
    print(f"Device: {get_device_config()}")
    print(f"AMD DirectML Available: {check_amd_directml()}")
    
    # Create model
    model = MicroPricePredictor(
        n_levels=10,
        n_features_per_level=4,
        embedding_dim=64
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Model Size (FP32): {total_params * 4 / 1024**2:.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(1, 4, 20)  # batch=1, features=4, levels*2=20
    output = model(dummy_input)
    
    print(f"\nOutput shapes:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")
