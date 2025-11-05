import os
import subprocess
import sys

def setup_project():
    print("🚀 Setting up SmartSpaces...")
    
    # Create directories
    directories = ['uploads', 'static', 'templates']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Install requirements
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except:
        print("❌ Error installing from requirements.txt")
    
    print("\n🎉 Setup complete! Run: python app.py")

if __name__ == "__main__":
    setup_project()