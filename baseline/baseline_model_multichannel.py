"""
BASELINE MODEL - MULTI-CHANNEL VERSION
ResNet50 adapted to accept 12 input channels instead of 3.

Channels:
- 0-2: RGB
- 3: DEM Height
- 4-11: Land Use (one-hot encoded)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class MultiChannelResNet50(nn.Module):
    """
    ResNet50 modified to accept 12-channel input.
    
    Strategy:
    - Replace first conv layer (3 -> 12 channels)
    - Initialize RGB channels with ImageNet pretrained weights
    - Initialize new channels (DEM + LandUse) with Kaiming initialization
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Load standard ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # === CRITICAL MODIFICATION: EXPAND FIRST CONV LAYER ===
        original_conv1 = self.backbone.conv1
        
        # Create new conv1 with 12 input channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels=12,  # RGB (3) + DEM (1) + LandUse (8) = 12
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Initialize weights strategically
        with torch.no_grad():
            if pretrained:
                # Copy pretrained RGB weights to first 3 channels
                self.backbone.conv1.weight[:, :3, :, :] = original_conv1.weight
                
                # Initialize remaining 9 channels (DEM + LandUse)
                # Strategy: Small random values so they don't dominate initially
                nn.init.kaiming_normal_(
                    self.backbone.conv1.weight[:, 3:, :, :],
                    mode='fan_out',
                    nonlinearity='relu'
                )
                
                # Scale down new channels to 10% of RGB importance initially
                # This allows gradual learning without overwhelming pretrained features
                self.backbone.conv1.weight[:, 3:, :, :] *= 0.1
            else:
                # If not using pretrained, initialize all channels uniformly
                nn.init.kaiming_normal_(
                    self.backbone.conv1.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
        
        # === REGRESSION HEAD ===
        # Replace classification head with regression output
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # Single output: population prediction
        )
        
        # === OPTIONAL: FREEZE BACKBONE ===
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:  # Don't freeze the regression head
                    param.requires_grad = False
            print("✅ Backbone frozen (only regression head trainable)")
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 12, 224, 224)
        
        Returns:
            Tensor of shape (B, 1) - population predictions
        """
        return self.backbone(x)


class DualBranchModel(nn.Module):
    """
    Advanced model combining:
    - CNN branch (multi-channel ResNet50)
    - Tabular features branch (MLP)
    
    Useful for incorporating road_density, building_density, poi_density.
    """
    
    def __init__(
        self, 
        pretrained: bool = True, 
        freeze_backbone: bool = False,
        tabular_dim: int = 3
    ):
        super().__init__()
        
        # CNN branch (multi-channel)
        self.cnn = MultiChannelResNet50(
            pretrained=pretrained, 
            freeze_backbone=freeze_backbone
        )
        
        # Tabular features branch
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1 + 16, 64),  # CNN output (1) + Tabular features (16)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x_image, x_tabular):
        """
        Args:
            x_image: (B, 12, 224, 224) - multi-channel satellite crop
            x_tabular: (B, 3) - road_density, building_density, poi_density
        
        Returns:
            (B, 1) - population prediction
        """
        # CNN branch
        cnn_out = self.cnn(x_image)  # (B, 1)
        
        # Tabular branch
        tab_out = self.tabular_mlp(x_tabular)  # (B, 16)
        
        # Concatenate and fuse
        combined = torch.cat([cnn_out, tab_out], dim=1)  # (B, 17)
        output = self.fusion(combined)  # (B, 1)
        
        return output


def get_model(
    model_type: str = 'baseline',
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'baseline' or 'dual_branch'
        pretrained: Use ImageNet pretrained weights for RGB channels
        freeze_backbone: Freeze all layers except regression head
        device: 'cuda' or 'cpu'
    
    Returns:
        Model moved to device
    """
    
    if model_type == 'baseline':
        model = MultiChannelResNet50(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        print(f"\n✅ Created Multi-Channel ResNet50 Baseline")
        print(f"   - Input: (B, 12, 224, 224)")
        print(f"   - Output: (B, 1)")
        print(f"   - Pretrained: {pretrained}")
        print(f"   - Frozen backbone: {freeze_backbone}")
    
    elif model_type == 'dual_branch':
        model = DualBranchModel(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            tabular_dim=3
        )
        print(f"\n✅ Created Dual-Branch Model (CNN + Tabular)")
        print(f"   - Image input: (B, 12, 224, 224)")
        print(f"   - Tabular input: (B, 3)")
        print(f"   - Output: (B, 1)")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    return model


# ============================================================================
# MODEL TEST
# ============================================================================

if __name__ == "__main__":
    """Test model creation and forward pass"""
    
    print("=" * 80)
    print(" MULTI-CHANNEL MODEL TEST")
    print("=" * 80)
    
    # Test baseline model
    print("\n1. Testing Baseline Model...")
    model = get_model(model_type='baseline', pretrained=True, device='cpu')
    
    # Create dummy input
    dummy_input = torch.randn(4, 12, 224, 224)  # Batch of 4 samples
    
    print("\nForward pass:")
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.squeeze()}")
    
    # Test dual branch model
    print("\n" + "=" * 80)
    print("2. Testing Dual-Branch Model...")
    model_dual = get_model(model_type='dual_branch', pretrained=True, device='cpu')
    
    dummy_tabular = torch.randn(4, 3)  # Batch of 4 samples with 3 features
    
    print("\nForward pass:")
    with torch.no_grad():
        output_dual = model_dual(dummy_input, dummy_tabular)
    
    print(f"  Image input shape: {dummy_input.shape}")
    print(f"  Tabular input shape: {dummy_tabular.shape}")
    print(f"  Output shape: {output_dual.shape}")
    print(f"  Output values: {output_dual.squeeze()}")
    
    print("\n" + "=" * 80)
    print(" ALL TESTS PASSED!")
    print("=" * 80)