import os
import subprocess
import sys

def quick_fix():
    print("🔧 Running quick fix...")
    
    # Install missing scipy
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy==1.11.3"])
        print("✅ Installed scipy")
    except:
        print("❌ Failed to install scipy")
    
    # Create necessary directories
    directories = ['uploads', 'static', 'templates', 'models', 'datasets']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created {dir_name}/")
    
    print("\n🎉 Quick fix complete! Now run: python app.py")

if __name__ == "__main__":
    quick_fix()