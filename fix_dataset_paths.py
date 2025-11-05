import os
import shutil

def fix_dataset_structure():
    print("🔧 Fixing dataset structure...")
    
    # Check what we have
    base_path = "datasets"
    if not os.path.exists(base_path):
        print("❌ datasets folder not found!")
        return False
    
    print("📁 Current structure in datasets/:")
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')
    
    # Look for x and y folders
    x_path = None
    y_path = None
    
    for root, dirs, files in os.walk(base_path):
        if 'x' in dirs and 'y' in dirs:
            x_path = os.path.join(root, 'x')
            y_path = os.path.join(root, 'y')
            break
    
    if x_path and y_path:
        print(f"✅ Found x folder: {x_path}")
        print(f"✅ Found y folder: {y_path}")
        
        # Create the expected structure
        target_path = "datasets/floor-plan-dataset"
        os.makedirs(target_path, exist_ok=True)
        
        # Create symbolic links or copy files
        try:
            # On Windows, we'll create junctions or copy
            if not os.path.exists(os.path.join(target_path, 'x')):
                os.symlink(x_path, os.path.join(target_path, 'x'))
                print("✅ Created symlink for x folder")
            if not os.path.exists(os.path.join(target_path, 'y')):
                os.symlink(y_path, os.path.join(target_path, 'y'))
                print("✅ Created symlink for y folder")
        except:
            # If symlinks fail, copy the files
            print("📋 Copying files instead of symlinks...")
            import shutil
            if not os.path.exists(os.path.join(target_path, 'x')):
                shutil.copytree(x_path, os.path.join(target_path, 'x'))
                print("✅ Copied x folder")
            if not os.path.exists(os.path.join(target_path, 'y')):
                shutil.copytree(y_path, os.path.join(target_path, 'y'))
                print("✅ Copied y folder")
        
        return True
    else:
        print("❌ Could not find x and y folders in the dataset")
        print("Please make sure your dataset has 'x' and 'y' folders")
        return False

if __name__ == "__main__":
    fix_dataset_structure()