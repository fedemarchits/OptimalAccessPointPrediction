"""
EMERGENCY FIX - R² ANCORA NEGATIVO
✅ ResNet50 baseline (PIÙ STABILE)
✅ LR ultra-basso
"""

import sys
import os
import torch

sys.path.append(os.getcwd())

from train_multichannel_FIXED import train

print("=" * 80)
print(" 🆘 EMERGENCY FIX - R² NEGATIVO PERSISTENTE")
print("=" * 80)

# ============================================================================
# PATHS
# ============================================================================

JSON_FILE = r"/workspace/PopulationDataset/final_clustered_samples.json"
BASE_DIR = r"/workspace/PopulationDataset"

# ============================================================================
# MODEL
# ============================================================================

MODEL_TYPE = 'efficientnet'   #
PRETRAINED = True
FREEZE_BACKBONE = False

# ============================================================================
# HYPERPARAMETERS - ULTRA CONSERVATIVI
# ============================================================================

NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
PATIENCE = 30

# ============================================================================
# OUTPUT PATHS
# ============================================================================

CHECKPOINT_DIR = f'checkpoints_EMERGENCY_{MODEL_TYPE}'
LOG_DIR = f'runs/EMERGENCY_{MODEL_TYPE}'

# ============================================================================
# WEIGHTS & BIASES
# ============================================================================

USE_WANDB = True
WANDB_PROJECT = 'population-prediction-emergency-2'
WANDB_NAME = f'{MODEL_TYPE}_emergency_bs{BATCH_SIZE}_lr{LEARNING_RATE}'

# ============================================================================
# HARDWARE
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 2

# ============================================================================
# CONFIG SUMMARY
# ============================================================================

print(f"\n📊 Configuration:")
print(f"   Model: {MODEL_TYPE} (ResNet50)")
print(f"   Learning Rate: {LEARNING_RATE:.2e}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Freeze Backbone: {FREEZE_BACKBONE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Device: {DEVICE}")

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

try:
    print("\n" + "=" * 80)
    print(" 🚀 STARTING EMERGENCY TRAINING")
    print("=" * 80)

    train(
        # Data
        json_file=JSON_FILE,
        base_dir=BASE_DIR,

        # Model
        model_type=MODEL_TYPE,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,

        # Training
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        # Early stopping
        patience=PATIENCE,

        # Checkpointing
        checkpoint_dir=CHECKPOINT_DIR,
        save_every=5,

        # Logging
        log_dir=LOG_DIR,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_name=WANDB_NAME,

        # System
        num_workers=NUM_WORKERS,
        device=DEVICE,
        resume_from=None
    )

    print("\n" + "=" * 80)
    print(" ✅ EMERGENCY TRAINING COMPLETED!")
    print("=" * 80)

    print(f"\n📁 Check results:")
    print(f"   Best model: {CHECKPOINT_DIR}/best_model.pth")
    print(f"   History: {CHECKPOINT_DIR}/training_history.json")
    print(f"   Logs: {LOG_DIR}")

    print(f"\n🔄 Next steps:")
    print(f"   1. Check if R² is positive now")
    print(f"   2. If still frozen, unfreeze and continue:")
    print(f"      FREEZE_BACKBONE = False")
    print(f"      resume_from = '{CHECKPOINT_DIR}/checkpoint_epoch_20.pth'")

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

    print(f"\n💡 Troubleshooting:")
    print(f"   1. Se R² ancora negativo: riduci LR a 1e-6")
    print(f"   2. Se loss non scende: prova normalizzazione sqrt")
    print(f"   3. Se OOM: BATCH_SIZE = 8")

print("\n" + "=" * 80)
