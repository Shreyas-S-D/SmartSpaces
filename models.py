import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import ndimage
import math

# Import dataset processor
try:
    from dataset_processor import FloorPlanDatasetProcessor
except ImportError:
    print("⚠️  dataset_processor not available, using fallback methods")

# Room standards (in feet)
ROOM_STANDARDS = {
    'bedroom': {'min_area': 120, 'preferred_area': 200, 'min_width': 10, 'min_length': 12},
    'bathroom': {'min_area': 35, 'preferred_area': 50, 'min_width': 5, 'min_length': 7},
    'living_room': {'min_area': 200, 'preferred_area': 300, 'min_width': 12, 'min_length': 15},
    'kitchen': {'min_area': 100, 'preferred_area': 150, 'min_width': 8, 'min_length': 10},
    'hallway': {'min_width': 3, 'preferred_width': 4}
}

def calculate_room_stats(grid, total_area_sqft=0):
    """Calculate statistics with real dimensions"""
    stats = {
        'bedroom': 0,
        'bathroom': 0, 
        'living_area': 0,
        'kitchen': 0,
        'walls': int(np.sum(grid == 1)),
        'open_space': int(np.sum(grid == 0)),
        'room_areas': {},
        'total_utilizable_area': 0
    }
    
    # Count rooms and calculate areas using connected components
    for room_type, room_name in [(2, 'bedroom'), (3, 'living_area'), (4, 'bathroom'), (5, 'kitchen')]:
        room_mask = grid == room_type
        if np.any(room_mask):
            labeled, num_features = ndimage.label(room_mask)
            stats[room_name] = num_features
            
            # Calculate area for each room
            for i in range(1, num_features + 1):
                room_area = np.sum(labeled == i)
                stats['room_areas'][f'{room_name}_{i}'] = int(room_area)  # Convert to Python int
    
    # Calculate utilization
    total_cells = grid.size
    used_space = total_cells - stats['open_space']
    stats['utilization_rate'] = float(round(used_space / total_cells * 100, 1))
    
    # If total area is provided, convert to square feet
    if total_area_sqft > 0:
        cell_area = total_area_sqft / total_cells
        stats['total_utilizable_area'] = round(used_space * cell_area, 1)
        
        # Convert room areas to square feet
        for room_key, cell_area_val in stats['room_areas'].items():
            stats['room_areas'][room_key] = round(cell_area_val * cell_area, 1)
    
    return stats

class RealDatasetLoader:
    def __init__(self):
        self.training_data = self.load_training_data()
    
    def load_training_data(self):
        """Load or process training data"""
        processed_file = 'datasets/floor-plan-dataset/processed/training_data.json'
        
        if os.path.exists(processed_file):
            try:
                with open(processed_file, 'r') as f:
                    return json.load(f)
            except:
                print("❌ Error loading training data, using empty dataset")
                return []
        else:
            print("📝 No processed dataset found, using empty dataset")
            return []
    
    def get_training_samples(self, count=100):
        """Get training samples for model training"""
        samples = []
        for sample in self.training_data[:count]:
            layout = np.array(sample['layout'])
            requirements = sample['requirements']
            samples.append((layout, requirements))
        return samples

class RuleBasedEngine:
    def __init__(self):
        self.room_sizes = {
            'bedroom': {'min': 12, 'preferred': 16},
            'bathroom': {'min': 6, 'preferred': 8},
            'living': {'min': 20, 'preferred': 25}
        }
    
    def apply_rules(self, layout, requirements):
        """Apply architectural rules to optimize layout"""
        height, width = layout.shape
        optimized = layout.copy().astype(np.float64)
        
        # Ensure border walls
        optimized[0, :] = 1
        optimized[-1, :] = 1
        optimized[:, 0] = 1
        optimized[:, -1] = 1
        
        bedrooms = requirements.get('bedrooms', 1)
        bathrooms = requirements.get('bathrooms', 1)
        
        # Professional room placement
        self.place_bedrooms(optimized, bedrooms)
        self.place_bathrooms(optimized, bathrooms)
        self.place_living_area(optimized)
        self.place_kitchen(optimized)
        self.add_hallways(optimized)
        
        return optimized.astype(np.int32)
    
    def place_bedrooms(self, layout, count):
        """Place bedrooms professionally"""
        height, width = layout.shape
        
        if count == 1:
            # Single bedroom (master)
            start_row = height // 4
            end_row = 3 * height // 4
            start_col = 2
            end_col = width // 3
            if end_row > start_row and end_col > start_col:
                layout[start_row:end_row, start_col:end_col] = 2
            
        elif count >= 2:
            # Multiple bedrooms
            room_height = max(4, (height - 4) // count)
            for i in range(count):
                start_row = 2 + i * (room_height + 1)
                end_row = min(start_row + room_height, height - 2)
                start_col = 2
                end_col = width // 3
                
                if end_row > start_row and end_col > start_col:
                    layout[start_row:end_row, start_col:end_col] = 2
    
    def place_bathrooms(self, layout, count):
        """Place bathrooms professionally"""
        height, width = layout.shape
        
        for i in range(count):
            if i == 0:
                # Attached to master bedroom
                start_row = height // 4
                end_row = min(start_row + 6, height - 2)
                start_col = width // 3 + 1
                end_col = min(start_col + 5, width - 2)
            else:
                # Additional bathrooms
                start_row = 2 + i * 6
                end_row = min(start_row + 5, height - 2)
                start_col = width - 6
                end_col = width - 1
            
            if end_row > start_row and end_col > start_col:
                layout[start_row:end_row, start_col:end_col] = 4
    
    def place_living_area(self, layout):
        """Place living area"""
        height, width = layout.shape
        
        start_row = max(2, height // 4)
        end_row = min(3 * height // 4, height - 2)
        start_col = width // 3
        end_col = 2 * width // 3
        
        if end_row > start_row and end_col > start_col:
            layout[start_row:end_row, start_col:end_col] = 3
    
    def place_kitchen(self, layout):
        """Place kitchen"""
        height, width = layout.shape
        
        start_row = max(2, height - 8)
        end_row = height - 2
        start_col = max(2, width - 10)
        end_col = width - 2
        
        if end_row > start_row and end_col > start_col:
            layout[start_row:end_row, start_col:end_col] = 5
    
    def add_hallways(self, layout):
        """Add connecting hallways"""
        height, width = layout.shape
        
        # Main hallway
        hall_row = height // 2
        if 0 < hall_row < height - 1:
            start_col = max(1, width // 3)
            end_col = min(2 * width // 3, width - 1)
            layout[hall_row, start_col:end_col] = 0

class DimensionAwareGenerator:
    def __init__(self):
        self.room_standards = ROOM_STANDARDS
        self.grid_size = 64
        self.dataset_loader = RealDatasetLoader()
        
    def generate_from_dimensions(self, length_ft, width_ft, requirements):
        """Generate floor plan based on real dimensions using learned patterns"""
        total_area = length_ft * width_ft
        
        # Calculate grid dimensions (1 cell = 1 foot)
        grid_length = min(self.grid_size, length_ft)
        grid_width = min(self.grid_size, width_ft)
        
        # Get similar layouts from dataset
        similar_layouts = self.find_similar_layouts(requirements, total_area)
        
        if similar_layouts:
            # Use learned patterns from dataset
            layout = self.adapt_dataset_layout(similar_layouts[0], grid_width, grid_length, requirements)
        else:
            # Fallback to rule-based generation
            layout = self.place_rooms_with_dimensions(grid_width, grid_length, requirements)
        
        return layout, total_area
    
    def find_similar_layouts(self, requirements, total_area):
        """Find similar floor plans from the dataset"""
        samples = self.dataset_loader.get_training_samples()
        similar = []
        
        target_bedrooms = requirements.get('bedrooms', 2)
        target_bathrooms = requirements.get('bathrooms', 1)
        
        for layout, sample_req in samples:
            sample_bedrooms = sample_req.get('bedrooms', 0)
            sample_bathrooms = sample_req.get('bathrooms', 0)
            
            # Simple similarity scoring
            bedroom_match = abs(sample_bedrooms - target_bedrooms) <= 1
            bathroom_match = abs(sample_bathrooms - target_bathrooms) <= 1
            
            if bedroom_match and bathroom_match:
                similar.append(layout)
        
        return similar[:5]  # Return top 5 matches
    
    def adapt_dataset_layout(self, dataset_layout, target_width, target_length, requirements):
        """Adapt a dataset layout to target dimensions"""
        # Resize to target dimensions
        adapted = cv2.resize(dataset_layout, (target_length, target_width), interpolation=cv2.INTER_NEAREST)
        
        # Ensure requirements are met
        adapted = self.adjust_room_counts(adapted, requirements)
        
        return adapted
    
    def adjust_room_counts(self, layout, requirements):
        """Adjust room counts to match requirements"""
        current_stats = calculate_room_stats(layout)
        
        # Adjust bedrooms
        bedroom_diff = requirements.get('bedrooms', 1) - current_stats['bedroom']
        if bedroom_diff > 0:
            layout = self.add_rooms(layout, 2, bedroom_diff)
        elif bedroom_diff < 0:
            layout = self.remove_rooms(layout, 2, -bedroom_diff)
        
        # Adjust bathrooms
        bathroom_diff = requirements.get('bathrooms', 1) - current_stats['bathroom']
        if bathroom_diff > 0:
            layout = self.add_rooms(layout, 4, bathroom_diff)
        elif bathroom_diff < 0:
            layout = self.remove_rooms(layout, 4, -bathroom_diff)
        
        return layout
    
    def add_rooms(self, layout, room_type, count):
        """Add rooms of specific type"""
        for _ in range(count):
            empty_spots = np.argwhere(layout == 0)
            if len(empty_spots) > 0:
                spot = empty_spots[0]
                # Create room of appropriate size
                room_size = 4 if room_type == 4 else 6  # Bathrooms smaller
                start_row = max(0, spot[0]-room_size//2)
                end_row = min(layout.shape[0], spot[0]+room_size//2)
                start_col = max(0, spot[1]-room_size//2)
                end_col = min(layout.shape[1], spot[1]+room_size//2)
                
                if end_row > start_row and end_col > start_col:
                    layout[start_row:end_row, start_col:end_col] = room_type
        
        return layout
    
    def remove_rooms(self, layout, room_type, count):
        """Remove rooms of specific type"""
        room_mask = layout == room_type
        if np.any(room_mask):
            labeled, num_features = ndimage.label(room_mask)
            
            # Remove smallest rooms first
            room_sizes = []
            for i in range(1, num_features + 1):
                size = np.sum(labeled == i)
                room_sizes.append((i, size))
            
            # Sort by size (smallest first)
            room_sizes.sort(key=lambda x: x[1])
            
            # Remove the smallest ones
            for i in range(min(count, len(room_sizes))):
                room_id = room_sizes[i][0]
                layout[labeled == room_id] = 0
        
        return layout
    
    def place_rooms_with_dimensions(self, grid_width, grid_length, requirements):
        """Rule-based room placement (fallback)"""
        grid = np.zeros((grid_width, grid_length), dtype=int)
        
        # Add border walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        bedrooms = requirements.get('bedrooms', 1)
        bathrooms = requirements.get('bathrooms', 1)
        
        # Simple room placement
        if bedrooms > 0:
            bed_width = min(8, grid_length // 3)
            bed_height = min(6, grid_width // max(2, bedrooms))
            
            for i in range(bedrooms):
                start_row = 2 + i * (bed_height + 1)
                end_row = start_row + bed_height
                start_col = 2
                end_col = start_col + bed_width
                
                if end_row < grid_width - 2 and end_col < grid_length - 2:
                    grid[start_row:end_row, start_col:end_col] = 2
        
        if bathrooms > 0:
            for i in range(bathrooms):
                start_row = 2 + i * 6
                end_row = start_row + 4
                start_col = grid_length - 6
                end_col = grid_length - 2
                
                if end_row < grid_width - 2:
                    grid[start_row:end_row, start_col:end_col] = 4
        
        # Living area in remaining space
        living_start_col = 10 if bedrooms > 0 else 2
        living_end_col = grid_length - 7 if bathrooms > 0 else grid_length - 2
        
        if living_end_col > living_start_col:
            grid[2:grid_width-2, living_start_col:living_end_col] = 3
        
        return grid

class ProfessionalLayoutGenerator:
    def __init__(self):
        self.dimension_generator = DimensionAwareGenerator()
        self.rule_engine = RuleBasedEngine()
    
    def generate_layout(self, requirements, input_type='dimensions', dimensions=None, input_layout=None):
        """Generate professional floor plan layout"""
        if input_type == 'dimensions' and dimensions:
            length_ft, width_ft = dimensions
            return self.dimension_generator.generate_from_dimensions(length_ft, width_ft, requirements)
        elif input_layout is not None:
            # Enhance existing layout
            return self.enhance_existing_layout(input_layout, requirements)
        else:
            # Generate from requirements only
            return self.generate_from_requirements(requirements)
    
    def enhance_existing_layout(self, layout, requirements):
        """Enhance an existing layout"""
        # Simple enhancement - resize and adjust
        enhanced = cv2.resize(layout, (64, 64), interpolation=cv2.INTER_NEAREST)
        return self.rule_engine.apply_rules(enhanced, requirements), 0  # 0 area for unknown
    
    def generate_from_requirements(self, requirements):
        """Generate layout from requirements only"""
        # Use default dimensions
        layout, area = self.dimension_generator.generate_from_dimensions(40, 30, requirements)
        return layout, area

# Initialize professional generator
professional_generator = ProfessionalLayoutGenerator()