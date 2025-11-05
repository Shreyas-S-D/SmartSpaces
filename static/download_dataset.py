import requests
import os
import zipfile
import gdown

def download_dataset():
    """Download a floor plan dataset"""
    print("📥 Downloading floor plan dataset...")
    
    # Create datasets directory
    os.makedirs('datasets', exist_ok=True)
    
    # Option 1: Download from Google Drive (small dataset)
    try:
        # This is a small sample dataset of floor plans
        url = "https://drive.google.com/uc?id=1L2I6MH_1A2XOlpO5p_mB7_NDzt3eJZQy"
        output = "datasets/floorplans.zip"
        
        gdown.download(url, output, quiet=False)
        
        # Extract
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('datasets/')
        
        print("✅ Dataset downloaded and extracted!")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("📝 Using synthetic data instead...")
        return False

if __name__ == "__main__":
    download_dataset()