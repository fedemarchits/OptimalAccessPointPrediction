"""
BASELINE MODEL - MULTI-CHANNEL VERSION - EXTENDED
✅ ResNet50 originale (baseline)
✅ EfficientNet-B3 (RACCOMANDATO - migliore performance)
✅ ConvNeXt-Tiny (PIÙ POTENTE - architettura moderna)
✅ Dual Branch models
✅ 100% compatibile con codice esistente

Channels:
- 0-2: RGB
- 3: DEM Height
- 4-11: Land Use (one-hot encoded)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

# Try to import timm for advanced models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️ timm not installed - EfficientNet and ConvNeXt not available")
    print("   Install with: pip install timm")


# ============================================================================
# CHANNEL ATTENTION MODULE
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention mechanism - learns importance of each channel
    Useful because not all channels are equally informative
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Global pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # Compute attention weights
        avg_weights = self.fc(avg_out)
        max_weights = self.fc(max_out)
        
        # Combine and normalize
        weights = self.sigmoid(avg_weights + max_weights).view(b, c, 1, 1)
        
        # Apply attention
        return x * weights


# ============================================================================
# ORIGINAL RESNET50 MODEL (BASELINE)
# ============================================================================

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
                nn.init.kaiming_normal_(
                    self.backbone.conv1.weight[:, 3:, :, :],
                    mode='fan_out',
                    nonlinearity='relu'
                )
                
                # Scale down new channels to 10% of RGB importance initially
                self.backbone.conv1.weight[:, 3:, :, :] *= 0.1
            else:
                nn.init.kaiming_normal_(
                    self.backbone.conv1.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
        
        # === REGRESSION HEAD ===
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # === OPTIONAL: FREEZE BACKBONE ===
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
            print("✅ Backbone frozen (only regression head trainable)")
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# EFFICIENTNET MODEL (RECOMMENDED)
# ============================================================================

if TIMM_AVAILABLE:
    class EfficientNetMultiChannel(nn.Module):
        """
        EfficientNet-B3 adapted for 12 channels.
        
        ✅ Advantages vs ResNet50:
        - 40% more efficient (fewer params, faster)
        - Better performance on vision tasks
        - Balanced scaling (depth + width + resolution)
        """
        
        def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
            super().__init__()
            
            # Load EfficientNet-B3
            self.backbone = timm.create_model(
                'efficientnet_b3',
                pretrained=pretrained,
                in_chans=3,
                num_classes=0,
                global_pool=''
            )
            
            # === MODIFY FIRST CONV LAYER ===
            original_conv = self.backbone.conv_stem
            
            self.backbone.conv_stem = nn.Conv2d(
                in_channels=12,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Strategic weight initialization
            with torch.no_grad():
                if pretrained:
                    # Copy RGB pretrained weights
                    self.backbone.conv_stem.weight[:, :3, :, :] = original_conv.weight
                    
                    # Initialize extra channels
                    nn.init.kaiming_normal_(
                        self.backbone.conv_stem.weight[:, 3:, :, :],
                        mode='fan_out',
                        nonlinearity='relu'
                    )
                    # Scale down to 10%
                    self.backbone.conv_stem.weight[:, 3:, :, :] *= 0.1
                else:
                    nn.init.kaiming_normal_(
                        self.backbone.conv_stem.weight,
                        mode='fan_out',
                        nonlinearity='relu'
                    )
            
            # === CHANNEL ATTENTION ===
            self.channel_attention = ChannelAttention(12)
            
            # === REGRESSION HEAD ===
            num_features = self.backbone.num_features  # 1536 for B3
            
            self.regression_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            
            # Freeze backbone if requested
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print("✅ EfficientNet backbone frozen")
        
        def forward(self, x):
            # Apply channel attention
            x = self.channel_attention(x)
            
            # Extract features
            features = self.backbone(x)
            
            # Regression
            output = self.regression_head(features)
            
            return output


# ============================================================================
# CONVNEXT MODEL (MOST POWERFUL)
# ============================================================================

if TIMM_AVAILABLE:
    class ConvNeXtMultiChannel(nn.Module):
        """
        ConvNeXt-Tiny - modern architecture competing with Transformers.
        
        ✅ Advantages:
        - State-of-the-art performance
        - Better generalization
        - More robust than ResNet/EfficientNet
        """
        
        def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
            super().__init__()
            
            # ConvNeXt-Tiny
            self.backbone = timm.create_model(
                'convnext_tiny',
                pretrained=pretrained,
                in_chans=3,
                num_classes=0,
                global_pool=''
            )
            
            # Modify first layer for 12 channels
            original_conv = self.backbone.stem[0]
            
            self.backbone.stem[0] = nn.Conv2d(
                12,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Weight initialization
            with torch.no_grad():
                if pretrained:
                    self.backbone.stem[0].weight[:, :3, :, :] = original_conv.weight
                    nn.init.kaiming_normal_(
                        self.backbone.stem[0].weight[:, 3:, :, :],
                        mode='fan_out',
                        nonlinearity='relu'
                    )
                    self.backbone.stem[0].weight[:, 3:, :, :] *= 0.1
                else:
                    nn.init.kaiming_normal_(
                        self.backbone.stem[0].weight,
                        mode='fan_out',
                        nonlinearity='relu'
                    )
            
            # Channel attention
            self.channel_attention = ChannelAttention(12)
            
            # Regression head
            num_features = self.backbone.num_features  # 768 for tiny
            
            self.regression_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(num_features),
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print("✅ ConvNeXt backbone frozen")
        
        def forward(self, x):
            x = self.channel_attention(x)
            features = self.backbone(x)
            output = self.regression_head(features)
            return output


# ============================================================================
# DUAL BRANCH MODELS
# ============================================================================

class DualBranchModel(nn.Module):
    """
    Original Dual Branch with ResNet50
    """
    
    def __init__(
        self, 
        pretrained: bool = True, 
        freeze_backbone: bool = False,
        tabular_dim: int = 3
    ):
        super().__init__()
        
        self.cnn = MultiChannelResNet50(
            pretrained=pretrained, 
            freeze_backbone=freeze_backbone
        )
        
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(1 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x_image, x_tabular):
        cnn_out = self.cnn(x_image)
        tab_out = self.tabular_mlp(x_tabular)
        combined = torch.cat([cnn_out, tab_out], dim=1)
        output = self.fusion(combined)
        return output


if TIMM_AVAILABLE:
    class ImprovedDualBranch(nn.Module):
        """
        Improved Dual Branch with EfficientNet
        """
        
        def __init__(
            self,
            pretrained: bool = True,
            freeze_backbone: bool = False,
            tabular_dim: int = 3
        ):
            super().__init__()
            
            self.cnn = EfficientNetMultiChannel(
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
            
            self.tabular_mlp = nn.Sequential(
                nn.Linear(tabular_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU()
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(1 + 16, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
        
        def forward(self, x_image, x_tabular):
            cnn_out = self.cnn(x_image)
            tab_out = self.tabular_mlp(x_tabular)
            combined = torch.cat([cnn_out, tab_out], dim=1)
            output = self.fusion(combined)
            return output


# ============================================================================
# FACTORY FUNCTION (COMPATIBLE WITH EXISTING CODE)
# ============================================================================

def get_model(
    model_type: str = 'baseline',
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 
            - 'baseline' - ResNet50 (original)
            - 'efficientnet' - EfficientNet-B3 (RECOMMENDED)
            - 'convnext' - ConvNeXt-Tiny (MOST POWERFUL)
            - 'dual_branch' - ResNet50 + Tabular
            - 'dual_branch_improved' - EfficientNet + Tabular
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze all layers except regression head
        device: 'cuda' or 'cpu'
    
    Returns:
        Model moved to device
    """
    
    print(f"\n{'='*80}")
    print(f" MODEL CREATION: {model_type.upper()}")
    print(f"{'='*80}")
    
    if model_type == 'baseline':
        model = MultiChannelResNet50(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        print("✅ Multi-Channel ResNet50 (Original Baseline)")
        
    elif model_type == 'efficientnet':
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for EfficientNet.\n"
                "Install with: pip install timm"
            )
        model = EfficientNetMultiChannel(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        print("✅ EfficientNet-B3 Multi-Channel (RECOMMENDED)")
        print("   📈 ~40% more efficient than ResNet50")
        print("   🎯 Better performance on vision tasks")
        
    elif model_type == 'convnext':
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for ConvNeXt.\n"
                "Install with: pip install timm"
            )
        model = ConvNeXtMultiChannel(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        print("✅ ConvNeXt-Tiny Multi-Channel (MOST POWERFUL)")
        print("   📈 Modern architecture (2022)")
        print("   🎯 Competes with Vision Transformers")
        
    elif model_type == 'dual_branch':
        model = DualBranchModel(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            tabular_dim=3
        )
        print("✅ Dual-Branch Model (ResNet50 + Tabular)")
        
    elif model_type == 'dual_branch_improved':
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for Improved Dual Branch.\n"
                "Install with: pip install timm"
            )
        model = ImprovedDualBranch(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            tabular_dim=3
        )
        print("✅ Improved Dual-Branch (EfficientNet + Tabular)")
    
    else:
        available = ['baseline', 'dual_branch']
        if TIMM_AVAILABLE:
            available.extend(['efficientnet', 'convnext', 'dual_branch_improved'])
        
        raise ValueError(
            f"Unknown model type: {model_type}\n"
            f"Available: {', '.join(available)}\n"
            f"Note: Install timm for advanced models: pip install timm"
        )
    
    model = model.to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Input: (B, 12, 224, 224)")
    print(f"   Output: (B, 1)")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Pretrained: {pretrained}")
    print(f"   Frozen backbone: {freeze_backbone}")
    print(f"   Device: {device}")
    
    return model


# ============================================================================
# MODEL TEST
# ============================================================================

if __name__ == "__main__":
    """Test model creation and forward pass"""
    
    print("=" * 80)
    print(" MULTI-CHANNEL MODEL TEST - EXTENDED")
    print("=" * 80)
    
    # Test baseline model
    print("\n1. Testing Baseline Model (ResNet50)...")
    model = get_model(model_type='baseline', pretrained=False, device='cpu')
    
    dummy_input = torch.randn(4, 12, 224, 224)
    
    print("\nForward pass:")
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✅ PASSED")
    
    # Test advanced models if available
    if TIMM_AVAILABLE:
        print("\n" + "=" * 80)
        print("2. Testing EfficientNet-B3...")
        model2 = get_model(model_type='efficientnet', pretrained=False, device='cpu')
        
        with torch.no_grad():
            output2 = model2(dummy_input)
        
        print(f"  Output shape: {output2.shape}")
        print(f"  ✅ PASSED")
        
        print("\n" + "=" * 80)
        print("3. Testing ConvNeXt-Tiny...")
        model3 = get_model(model_type='convnext', pretrained=False, device='cpu')
        
        with torch.no_grad():
            output3 = model3(dummy_input)
        
        print(f"  Output shape: {output3.shape}")
        print(f"  ✅ PASSED")
    else:
        print("\n⚠️ Skipping advanced models (timm not installed)")
        print("   Install with: pip install timm")
    
    print("\n" + "=" * 80)
    print(" ALL TESTS PASSED!")
    print("=" * 80)
