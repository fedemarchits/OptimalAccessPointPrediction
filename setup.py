"""
SETUP SCRIPT - INSTALLAZIONE RAPIDA
Prepara l'ambiente per i modelli migliorati
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Esegue un comando e mostra il risultato"""
    print(f"\n{'='*80}")
    print(f"📦 {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

def check_installation(package_name, import_name=None):
    """Verifica se un package è installato"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def main():
    print("="*80)
    print(" 🚀 SETUP SCRIPT - MODELLI MIGLIORATI")
    print("="*80)
    
    # 1. Check Python version
    print(f"\n📋 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  WARNING: Python 3.8+ raccomandato")
    
    # 2. Check existing packages
    print("\n" + "="*80)
    print(" 🔍 CHECKING EXISTING PACKAGES")
    print("="*80)
    
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'timm': 'timm',
        'wandb': 'wandb',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'rasterio': 'rasterio',
        'PIL': 'PIL',
        'tqdm': 'tqdm',
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        if check_installation(package, import_name):
            print(f"   ✅ {package}")
        else:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    # 3. Install missing packages
    if missing_packages:
        print("\n" + "="*80)
        print(" 📦 INSTALLING MISSING PACKAGES")
        print("="*80)
        
        print(f"\nMissing: {', '.join(missing_packages)}")
        
        response = input("\nDo you want to install them? (y/n): ")
        
        if response.lower() == 'y':
            # Install core packages
            if 'timm' in missing_packages:
                run_command(
                    f"{sys.executable} -m pip install timm",
                    "Installing timm (PyTorch Image Models)"
                )
            
            if 'wandb' in missing_packages:
                run_command(
                    f"{sys.executable} -m pip install wandb",
                    "Installing Weights & Biases"
                )
            
            # Install other missing packages
            other_packages = [p for p in missing_packages if p not in ['timm', 'wandb']]
            if other_packages:
                run_command(
                    f"{sys.executable} -m pip install {' '.join(other_packages)}",
                    f"Installing {', '.join(other_packages)}"
                )
        else:
            print("\n⚠️  Installation skipped")
    else:
        print("\n✅ All required packages already installed!")
    
    # 4. Check CUDA availability
    print("\n" + "="*80)
    print(" 💻 CHECKING CUDA/GPU")
    print("="*80)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available!")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA not available - training will use CPU")
            print("   Training on CPU will be MUCH slower")
            print("   Consider using Google Colab or a GPU-enabled machine")
    except ImportError:
        print("❌ PyTorch not installed")
    
    # 5. Test model creation
    print("\n" + "="*80)
    print(" 🧪 TESTING MODEL CREATION")
    print("="*80)
    
    try:
        from improved_model_satellite import get_model
        
        print("\nTesting EfficientNet creation...")
        model = get_model('efficientnet', pretrained=False, device='cpu')
        print("✅ EfficientNet OK")
        
        print("\nTesting ConvNeXt creation...")
        model = get_model('convnext', pretrained=False, device='cpu')
        print("✅ ConvNeXt OK")
        
        print("\n✅ All models can be created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        print("\nTroubleshooting:")
        print("   1. Make sure timm is installed: pip install timm")
        print("   2. Check that improved_model_satellite.py is in the current directory")
    
    # 6. Create directories
    print("\n" + "="*80)
    print(" 📁 CREATING DIRECTORIES")
    print("="*80)
    
    directories = [
        'checkpoints_efficientnet',
        'checkpoints_convnext',
        'checkpoints_baseline',
        'runs'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✅ {dir_name}")
    
    # 7. Final summary
    print("\n" + "="*80)
    print(" ✅ SETUP COMPLETE!")
    print("="*80)
    
    print("\n📝 Next steps:")
    print("   1. Review GUIDA_MODELLI_MIGLIORATI.md for detailed guide")
    print("   2. (Optional) Run: python compare_models.py --device cuda")
    print("   3. Start training: python run_training_IMPROVED.py")
    
    print("\n💡 Quick start:")
    print("   # Compare models (optional)")
    print("   python compare_models.py")
    print()
    print("   # Train with EfficientNet (recommended)")
    print("   python run_training_IMPROVED.py")
    print()
    print("   # Or train with ConvNeXt (more powerful)")
    print("   # Edit run_training_IMPROVED.py: MODEL_TYPE = 'convnext'")
    print("   python run_training_IMPROVED.py")
    
    if 'wandb' in missing_packages or not check_installation('wandb'):
        print("\n📊 Weights & Biases setup (optional but recommended):")
        print("   wandb login")
        print("   # Then set USE_WANDB = True in run_training_IMPROVED.py")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
