import os

def check_dataset_files():
    print("🔍 Checking dataset files...")
    
    x_path = "datasets/floor-plan-dataset/x"
    y_path = "datasets/floor-plan-dataset/y"
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("❌ Dataset folders not found!")
        return
    
    x_files = sorted(os.listdir(x_path))
    y_files = sorted(os.listdir(y_path))
    
    print(f"📁 Files in x/ folder ({len(x_files)} files):")
    for f in x_files[:10]:  # Show first 10
        print(f"   {f}")
    if len(x_files) > 10:
        print(f"   ... and {len(x_files) - 10} more")
    
    print(f"📁 Files in y/ folder ({len(y_files)} files):")
    for f in y_files[:10]:
        print(f"   {f}")
    if len(y_files) > 10:
        print(f"   ... and {len(y_files) - 10} more")
    
    # Check if files match by name pattern
    print("\n🔍 Checking file matching...")
    
    # Try different matching strategies
    x_stems = [os.path.splitext(f)[0] for f in x_files]
    y_stems = [os.path.splitext(f)[0] for f in y_files]
    
    common_stems = set(x_stems) & set(y_stems)
    print(f"✅ Files with same names: {len(common_stems)}")
    
    if common_stems:
        print("Matching files:")
        for stem in list(common_stems)[:5]:
            print(f"   {stem}.*")
    
    # If no common names, try sequential matching
    if not common_stems and len(x_files) == len(y_files):
        print("🔄 Trying sequential matching...")
        matches = []
        for i in range(min(len(x_files), len(y_files))):
            matches.append((x_files[i], y_files[i]))
        
        print(f"✅ Sequential matches: {len(matches)}")
        for x, y in matches[:5]:
            print(f"   {x} -> {y}")
        
        return matches
    
    return list(common_stems)

if __name__ == "__main__":
    check_dataset_files()