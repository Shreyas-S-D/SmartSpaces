import cv2
import numpy as np
from skimage import measure, morphology

def process_image_to_layout(image_path, target_size=20):
    """Convert uploaded floor plan image to structured layout grid"""
    try:
        # Read and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for room detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create initial layout grid
        layout = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Scale factor for converting image coordinates to grid coordinates
        scale_x = target_size / img.shape[1]
        scale_y = target_size / img.shape[0]
        
        # Mark walls (edges)
        for y in range(edges.shape[0]):
            for x in range(edges.shape[1]):
                if edges[y, x] > 0:
                    grid_x = int(x * scale_x)
                    grid_y = int(y * scale_y)
                    if 0 <= grid_x < target_size and 0 <= grid_y < target_size:
                        layout[grid_y, grid_x] = 1  # Wall
        
        # Mark rooms from large contours
        min_room_area = (target_size * target_size) * 0.05  # 5% of total area
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area * scale_x * scale_y > min_room_area:
                # Create mask for this room
                mask = np.zeros_like(gray)
                cv2.fillPoly(mask, [contour], 255)
                
                # Convert room to grid coordinates
                for y in range(mask.shape[0]):
                    for x in range(mask.shape[1]):
                        if mask[y, x] > 0:
                            grid_x = int(x * scale_x)
                            grid_y = int(y * scale_y)
                            if (0 <= grid_x < target_size and 0 <= grid_y < target_size and 
                                layout[grid_y, grid_x] == 0):  # Not already a wall
                                layout[grid_y, grid_x] = 2  # Room
        
        # Ensure border walls
        layout[0, :] = 1
        layout[-1, :] = 1
        layout[:, 0] = 1
        layout[:, -1] = 1
        
        return layout
        
    except Exception as e:
        print(f"Image processing error: {e}")
        # Return default 20x20 layout on error
        default_layout = np.zeros((target_size, target_size), dtype=np.uint8)
        default_layout[0, :] = 1
        default_layout[-1, :] = 1
        default_layout[:, 0] = 1
        default_layout[:, -1] = 1
        return default_layout

def enhance_layout_quality(layout):
    """Post-process layout to improve quality"""
    # Remove small isolated pixels
    cleaned = morphology.remove_small_objects(layout.astype(bool), min_size=3)
    layout = layout * cleaned
    
    # Fill small holes in rooms
    layout = morphology.remove_small_holes(layout.astype(bool), area_threshold=2)
    
    return layout.astype(np.uint8)