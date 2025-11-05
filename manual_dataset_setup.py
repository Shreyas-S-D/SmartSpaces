import os
import shutil

def manual_dataset_setup():
    print("🔧 Manual Dataset Setup")
    
    # Ask user for the actual path to their dataset
    dataset_path = input("Enter the full path to your floor-plan-dataset folder (or press Enter for default): ").strip()
    
    if not dataset_path:
        # Try to find it automatically
        possible_paths = [
            "floor-plan-dataset",
            "datasets/floor-plan-dataset", 
            "../floor-plan-dataset",
            "./floor-plan-dataset"
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'x')):
                dataset_path = path
                break
    
    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Could not find dataset folder automatically.")
        print("Please manually copy your dataset folder structure:")
        print("Smartspaces/")
        print("├── datasets/")
        print("│   └── floor-plan-dataset/")
        print("│       ├── x/")
        print("│       └── y/")
        return False
    
    print(f"📁 Using dataset from: {dataset_path}")
    
    # Create the expected structure
    target_path = "datasets/floor-plan-dataset"
    os.makedirs(target_path, exist_ok=True)
    
    # Copy or link the folders
    x_source = os.path.join(dataset_path, 'x')
    y_source = os.path.join(dataset_path, 'y')
    
    if os.path.exists(x_source) and os.path.exists(y_source):
        try:
            # Try symbolic links first
            if not os.path.exists(os.path.join(target_path, 'x')):
                os.symlink(os.path.abspath(x_source), os.path.join(target_path, 'x'))
            if not os.path.exists(os.path.join(target_path, 'y')):
                os.symlink(os.path.abspath(y_source), os.path.join(target_path, 'y'))
            print("✅ Created symbolic links")
        except:
            # Fallback to copying
            print("📋 Copying files (this may take a while)...")
            import shutil
            if not os.path.exists(os.path.join(target_path, 'x')):
                shutil.copytree(x_source, os.path.join(target_path, 'x'))
            if not os.path.exists(os.path.join(target_path, 'y')):
                shutil.copytree(y_source, os.path.join(target_path, 'y'))
            print("✅ Copied dataset files")
        
        # Verify
        x_files = len(os.listdir(os.path.join(target_path, 'x')))
        y_files = len(os.listdir(os.path.join(target_path, 'y')))
        print(f"✅ Dataset ready: {x_files} images in x/, {y_files} images in y/")
        return True
    else:
        print("❌ Could not find x and y folders in the specified path")
        return False

if __name__ == "__main__":
    manual_dataset_setup()