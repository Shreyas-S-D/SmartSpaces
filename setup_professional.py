import os
import subprocess
import sys

def setup_professional():
    print("🚀 Setting up Professional SmartSpaces...")
    
    # Create directories
    directories = ['uploads', 'static', 'templates', 'models', 'datasets']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Install requirements
    print("📦 Installing professional dependencies...")
    packages = [
        "Flask==2.3.3",
        "opencv-python==4.8.1.78",
        "torch",
        "torchvision", 
        "torchaudio",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "Pillow==10.0.1",
        "matplotlib==3.7.2",
        "scikit-learn==1.3.0",
        "scikit-image==0.21.0",
        "scipy==1.11.3",
        "gdown==4.7.1",
        "requests==2.31.0",
        "tqdm==4.66.1"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except:
            print(f"⚠️  Failed to install: {package}")
    
    print("\n🎉 Professional setup complete!")
    print("📝 Next steps:")
    print("1. Run: python app.py")
    print("2. The system will generate synthetic training data")
    print("3. Access: http://localhost:5000")
    print("4. For even better results, add real floor plan images to datasets/ folder")

if __name__ == "__main__":
    setup_professional()