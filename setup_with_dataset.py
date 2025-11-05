import os
import subprocess
import sys

def setup_with_dataset():
    print("🚀 Setting up SmartSpaces with Real Dataset...")
    
    # Create directories
    directories = ['uploads', 'static', 'templates', 'models', 'datasets']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Install requirements
    print("📦 Installing dependencies...")
    packages = [
        "Flask==2.3.3",
        "opencv-python==4.8.1.78",
        "torch",
        "torchvision", 
        "numpy==1.24.3",
        "pandas==2.0.3",
        "Pillow==10.0.1",
        "matplotlib==3.7.2",
        "scikit-learn==1.3.0",
        "scikit-image==0.21.0",
        "scipy==1.11.3",
        "tqdm==4.66.1"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except:
            print(f"⚠️  Failed to install: {package}")
    
    print("\n📁 Dataset Setup:")
    print("1. Your dataset should be in: datasets/floor-plan-dataset/")
    print("2. With folders: x/ and y/")
    print("3. Run: python dataset_processor.py (to process the dataset)")
    print("4. Then run: python app.py")
    print("5. Access: http://localhost:5000")

if __name__ == "__main__":
    setup_with_dataset()