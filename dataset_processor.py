import os
import cv2
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

class FloorPlanDatasetProcessor:
    def __init__(self, dataset_path='datasets/floor-plan-dataset'):
        self.dataset_path = dataset_path
        self.x_path = os.path.join(dataset_path, 'x')
        self.y_path = os.path.join(dataset_path, 'y')
        self.processed_path = os.path.join(dataset_path, 'processed')
        
        # Create processed directory
        os.makedirs(self.processed_path, exist_ok=True)
        
        # Verify paths exist
        if not os.path.exists(self.x_path):
            print(f"❌ Error: x folder not found at {self.x_path}")
            return
            
        if not os.path.exists(self.y_path):
            print(f"❌ Error: y folder not found at {self.y_path}")
            return
        
        print(f"✅ Dataset paths verified:")
        print(f"   x: {self.x_path} ({len(os.listdir(self.x_path))} files)")
        print(f"   y: {self.y_path} ({len(os.listdir(self.y_path))} files)")
        
        # Room color mapping (from the dataset)
        self.room_colors = {
            (0, 0, 0): 0,        # Background
            (192, 192, 224): 1,  # Closet
            (192, 255, 255): 2,  # Bathroom
            (224, 255, 192): 3,  # Living Room
            (255, 224, 128): 4,  # Bedroom
            (255, 160, 96): 5,   # Hall
            (255, 224, 224): 6,  # Balcony
            (224, 224, 224): 7,  # Not used
            (224, 224, 128): 8   # Not used
        }
        
        # Simplified room types for our system
        self.room_mapping = {
            1: 0,  # Closet -> Open Space
            2: 4,  # Bathroom -> Bathroom
            3: 3,  # Living Room -> Living Area
            4: 2,  # Bedroom -> Bedroom
            5: 0,  # Hall -> Open Space
            6: 0,  # Balcony -> Open Space
        }
    
    def get_file_pairs(self):
        """Get matching file pairs from x and y folders"""
        x_files = sorted(os.listdir(self.x_path))
        y_files = sorted(os.listdir(self.y_path))
        
        print(f"Found {len(x_files)} files in x/, {len(y_files)} files in y/")
        
        # Strategy 1: Exact name matching
        common_files = set(x_files) & set(y_files)
        if common_files:
            print(f"✅ Found {len(common_files)} exact name matches")
            return [(f, f) for f in common_files]
        
        # Strategy 2: Stem matching (without extension)
        x_stems = [os.path.splitext(f)[0] for f in x_files]
        y_stems = [os.path.splitext(f)[0] for f in y_files]
        
        common_stems = set(x_stems) & set(y_stems)
        if common_stems:
            print(f"✅ Found {len(common_stems)} stem matches")
            pairs = []
            for stem in common_stems:
                x_file = next(f for f in x_files if os.path.splitext(f)[0] == stem)
                y_file = next(f for f in y_files if os.path.splitext(f)[0] == stem)
                pairs.append((x_file, y_file))
            return pairs
        
        # Strategy 3: Sequential matching (same order)
        if len(x_files) == len(y_files):
            print(f"🔄 Using sequential matching for {len(x_files)} files")
            return list(zip(x_files, y_files))
        
        # Strategy 4: Try common patterns
        print("🔄 Trying pattern-based matching...")
        pairs = []
        
        # Common patterns in floor plan datasets
        patterns = [
            ('image_{}.png', 'mask_{}.png'),
            ('img_{}.jpg', 'label_{}.png'),
            ('{}.jpg', '{}.png'),
            ('{}.png', '{}.png'),
            ('input_{}.png', 'output_{}.png'),
        ]
        
        for x_pattern, y_pattern in patterns:
            try:
                matched_pairs = []
                for i in range(min(len(x_files), len(y_files))):
                    x_test = x_pattern.format(i+1)
                    y_test = y_pattern.format(i+1)
                    
                    if x_test in x_files and y_test in y_files:
                        matched_pairs.append((x_test, y_test))
                
                if matched_pairs:
                    print(f"✅ Found {len(matched_pairs)} matches with pattern: {x_pattern} -> {y_pattern}")
                    return matched_pairs
            except:
                continue
        
        print("❌ Could not find matching file pairs")
        return []
    
    def convert_to_json_serializable(self, obj):
        """Convert NumPy types to JSON serializable Python types"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def process_dataset(self, max_samples=100):
        """Process the entire dataset and extract room layouts"""
        print("🔄 Processing floor plan dataset...")
        
        try:
            file_pairs = self.get_file_pairs()
            
            if not file_pairs:
                print("❌ No file pairs found to process")
                return []
            
            print(f"Processing {min(len(file_pairs), max_samples)} file pairs...")
            
            processed_samples = []
            
            for x_file, y_file in tqdm(file_pairs[:max_samples]):
                try:
                    sample = self.process_sample(x_file, y_file)
                    if sample:
                        # Convert all NumPy types to Python types
                        sample = self.convert_to_json_serializable(sample)
                        processed_samples.append(sample)
                except Exception as e:
                    print(f"Error processing {x_file}/{y_file}: {e}")
                    continue
            
            # Save processed data
            with open(os.path.join(self.processed_path, 'training_data.json'), 'w') as f:
                json.dump(processed_samples, f, indent=2)
            
            print(f"✅ Successfully processed {len(processed_samples)} floor plans")
            return processed_samples
            
        except Exception as e:
            print(f"❌ Error processing dataset: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_sample(self, x_file, y_file):
        """Process a single floor plan sample"""
        # Load original image and mask
        img_path = os.path.join(self.x_path, x_file)
        mask_path = os.path.join(self.y_path, y_file)
        
        # Try different image reading methods
        img = self.read_image(img_path)
        mask = self.read_image(mask_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return None
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            return None
        
        print(f"✅ Loaded: {x_file} ({img.shape}), {y_file} ({mask.shape})")
        
        # Convert mask to our room format
        room_layout = self.mask_to_room_layout(mask)
        
        if room_layout is None:
            return None
            
        # Resize to standard size
        room_layout = cv2.resize(room_layout, (64, 64), interpolation=cv2.INTER_NEAREST)
        
        # Extract room requirements
        requirements = self.extract_requirements(room_layout)
        
        return {
            'x_file': x_file,
            'y_file': y_file,
            'layout': room_layout.tolist(),  # Convert to list for JSON
            'requirements': requirements,
            'original_size': [int(img.shape[0]), int(img.shape[1])]  # Convert to Python int
        }
    
    def read_image(self, path):
        """Read image with multiple fallbacks"""
        # Try OpenCV
        img = cv2.imread(path)
        if img is not None:
            return img
        
        # Try PIL as fallback
        try:
            pil_img = Image.open(path)
            return np.array(pil_img)
        except:
            pass
        
        # Try with different extensions
        base, ext = os.path.splitext(path)
        for new_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            if new_ext != ext:
                new_path = base + new_ext
                if os.path.exists(new_path):
                    img = cv2.imread(new_path)
                    if img is not None:
                        print(f"✅ Found {new_path} instead of {path}")
                        return img
        
        print(f"❌ Could not read image: {path}")
        return None
    
    def mask_to_room_layout(self, mask):
        """Convert color mask to room layout grid"""
        if mask is None:
            return None
            
        height, width = mask.shape[:2]
        layout = np.zeros((height, width), dtype=np.uint8)
        
        # Handle different color formats
        if len(mask.shape) == 2:  # Grayscale
            # Map grayscale to room types
            gray_ranges = {
                (0, 50): 0,      # Background
                (50, 100): 1,    # Closet
                (100, 150): 2,   # Bathroom
                (150, 200): 3,   # Living Room
                (200, 255): 4,   # Bedroom
            }
            
            for (low, high), room_id in gray_ranges.items():
                mask_range = (mask >= low) & (mask <= high)
                layout[mask_range] = room_id
                
        else:  # Color image
            for color, room_id in self.room_colors.items():
                # Try both BGR and RGB
                color_bgr = np.array(color[::-1])  # RGB to BGR
                color_rgb = np.array(color)        # RGB
                
                color_mask_bgr = np.all(mask == color_bgr, axis=-1)
                color_mask_rgb = np.all(mask == color_rgb, axis=-1)
                
                layout[color_mask_bgr | color_mask_rgb] = room_id
        
        # Map to our simplified room types
        for old_id, new_id in self.room_mapping.items():
            layout[layout == old_id] = new_id
        
        return layout
    
    def extract_requirements(self, layout):
        """Extract room requirements from layout - returns Python native types"""
        from scipy import ndimage
        
        requirements = {
            'bedrooms': 0,
            'bathrooms': 0,
            'living_room': False,  # Python bool, not numpy bool_
            'total_area': int(layout.size)  # Python int
        }
        
        # Count bedrooms using connected components
        bedroom_mask = layout == 2
        if np.any(bedroom_mask):
            labeled, num_features = ndimage.label(bedroom_mask)
            requirements['bedrooms'] = int(num_features)  # Convert to Python int
        
        # Count bathrooms
        bathroom_mask = layout == 4
        if np.any(bathroom_mask):
            labeled, num_features = ndimage.label(bathroom_mask)
            requirements['bathrooms'] = int(num_features)  # Convert to Python int
        
        # Check for living area - convert numpy bool to Python bool
        requirements['living_room'] = bool(np.any(layout == 3))
        
        return requirements
    
    def visualize_sample(self, x_file=None, y_file=None):
        """Visualize a processed sample"""
        if x_file is None or y_file is None:
            file_pairs = self.get_file_pairs()
            if file_pairs:
                x_file, y_file = file_pairs[0]
            else:
                print("No file pairs found")
                return
        
        img_path = os.path.join(self.x_path, x_file)
        mask_path = os.path.join(self.y_path, y_file)
        
        img = self.read_image(img_path)
        mask = self.read_image(mask_path)
        
        if img is None or mask is None:
            print(f"Could not load {x_file} or {y_file}")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(img.shape) == 3:
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'Original: {x_file}')
        axes[0].axis('off')
        
        # Original mask
        if len(mask.shape) == 3:
            axes[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Mask: {y_file}')
        axes[1].axis('off')
        
        # Processed layout
        layout = self.mask_to_room_layout(mask)
        if layout is not None:
            layout = cv2.resize(layout, (64, 64), interpolation=cv2.INTER_NEAREST)
            
            colors = ['white', 'black', 'lightblue', 'lightgreen', 'lightcoral']
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(colors)
            
            axes[2].imshow(layout, cmap=cmap, vmin=0, vmax=4)
            axes[2].set_title('Processed Layout')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Sample visualization saved as sample_visualization.png")

# Usage
if __name__ == "__main__":
    processor = FloorPlanDatasetProcessor()
    
    # First check what files we have
    file_pairs = processor.get_file_pairs()
    
    if file_pairs:
        print(f"✅ Found {len(file_pairs)} file pairs")
        # Process a small sample first
        samples = processor.process_dataset(max_samples=50)
        
        if samples:
            print(f"✅ Successfully processed {len(samples)} samples")
            # Visualize first sample
            processor.visualize_sample(samples[0]['x_file'], samples[0]['y_file'])
            
            # Show some statistics
            print("\n📊 Dataset Statistics:")
            total_bedrooms = sum(sample['requirements']['bedrooms'] for sample in samples)
            total_bathrooms = sum(sample['requirements']['bathrooms'] for sample in samples)
            has_living_room = sum(1 for sample in samples if sample['requirements']['living_room'])
            
            print(f"   Total samples: {len(samples)}")
            print(f"   Total bedrooms: {total_bedrooms}")
            print(f"   Total bathrooms: {total_bathrooms}")
            print(f"   Samples with living room: {has_living_room}")
            
        else:
            print("❌ No samples were processed")
    else:
        print("❌ No file pairs found")
        print("Trying to visualize any available files...")
        # Try to visualize whatever files exist
        x_files = os.listdir(processor.x_path)
        y_files = os.listdir(processor.y_path)
        if x_files and y_files:
            processor.visualize_sample(x_files[0], y_files[0])