from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import base64
import io
from PIL import Image
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from io import BytesIO
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Simple Rule-Based Engine
class RuleBasedEngine:
    def apply_rules(self, layout, requirements):
        height, width = layout.shape
        optimized = layout.copy()
        
        # Border walls
        optimized[0, :] = 1
        optimized[-1, :] = 1
        optimized[:, 0] = 1
        optimized[:, -1] = 1
        
        bedrooms = requirements.get('bedrooms', 2)
        bathrooms = requirements.get('bathrooms', 1)
        
        # Place bedrooms
        bed_height = min(6, height // max(2, bedrooms))
        for i in range(bedrooms):
            start_row = 2 + i * (bed_height + 1)
            end_row = min(start_row + bed_height, height - 2)
            start_col = 2
            end_col = min(start_col + 8, width - 2)
            optimized[start_row:end_row, start_col:end_col] = 2
        
        # Place bathrooms
        for i in range(bathrooms):
            start_row = 2 + i * 6
            end_row = min(start_row + 4, height - 2)
            start_col = width - 6
            end_col = width - 2
            optimized[start_row:end_row, start_col:end_col] = 4
        
        # Living area
        living_start_col = 12
        living_end_col = width - 8
        if living_end_col > living_start_col:
            optimized[2:height-2, living_start_col:living_end_col] = 3
        
        return optimized

def calculate_room_stats(grid, total_area=0):
    stats = {'bedroom': 0, 'bathroom': 0, 'living_area': 0, 'utilization_rate': 50}
    
    bedroom_mask = grid == 2
    if np.any(bedroom_mask):
        labeled, num_features = ndimage.label(bedroom_mask)
        stats['bedroom'] = num_features
    
    bathroom_mask = grid == 4
    if np.any(bathroom_mask):
        labeled, num_features = ndimage.label(bathroom_mask)
        stats['bathroom'] = num_features
    
    living_mask = grid == 3
    if np.any(living_mask):
        labeled, num_features = ndimage.label(living_mask)
        stats['living_area'] = num_features
    
    return stats

def grid_to_image(grid, option_num, total_area=0):
    try:
        colors = ['white', 'black', 'lightblue', 'lightgreen', 'lightcoral']
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, aspect='equal')
        ax.set_title(f'Floor Plan Option {option_num}', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add room labels
        for room_type, color, label in [(2, 'darkblue', 'Bedroom'), (3, 'darkgreen', 'Living'), (4, 'darkred', 'Bathroom')]:
            positions = np.argwhere(grid == room_type)
            if len(positions) > 0:
                center = positions.mean(axis=0)
                ax.text(center[1], center[0], label, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_base64}"
    except:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_plan():
    try:
        data = request.get_json()
        requirements = data.get('requirements', {})
        input_type = data.get('input_type')
        
        bedrooms = requirements.get('bedrooms', 2)
        bathrooms = requirements.get('bathrooms', 1)
        
        if input_type == 'dimensions':
            length = data.get('length', 20)
            width = data.get('width', 15)
            # Create base layout from dimensions
            grid = np.zeros((min(30, width), min(30, length)), dtype=int)
        else:
            # For photo input, create a simple default layout
            grid = np.zeros((20, 20), dtype=int)
        
        # Generate 3 options using rule-based engine
        engine = RuleBasedEngine()
        plans = []
        for i in range(3):
            plan = engine.apply_rules(grid.copy(), requirements)
            plans.append(plan)
        
        # Create response
        plans_with_images = []
        for i, plan in enumerate(plans):
            img_base64 = grid_to_image(plan, i + 1)
            room_stats = calculate_room_stats(plan)
            
            plans_with_images.append({
                'grid': plan.tolist(),
                'image': img_base64,
                'room_stats': room_stats
            })
        
        return jsonify({
            'success': True,
            'plans': plans_with_images,
            'message': 'Generated 3 floor plan options'
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)