import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

# Constants for grid detection and cell extraction
CELL_COUNT = 49  # 7x7 grid
MIN_CELL_AREA = 1000
GRID_DIVISION_FACTOR = 30
CELL_SIZE_TOLERANCE = 0.2
CELL_MARGIN_PERCENT = 0.05

def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

def detect_grid_lines(thresh: np.ndarray) -> np.ndarray:
    horizontal = np.copy(thresh)
    vertical = np.copy(thresh)
    
    # Horizontal lines
    cols = horizontal.shape[1]
    horizontal_size = cols // GRID_DIVISION_FACTOR
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    
    # Vertical lines
    rows = vertical.shape[0]
    vertical_size = rows // GRID_DIVISION_FACTOR
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    return cv2.add(horizontal, vertical)

def filter_cells(cells: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    if not cells:
        raise ValueError("No cells detected")
    
    avg_width = sum(w for _,_,w,_ in cells) / len(cells)
    avg_height = sum(h for _,_,_,h in cells) / len(cells)
    
    return [(x,y,w,h) for x,y,w,h in cells 
            if (1-CELL_SIZE_TOLERANCE <= w/avg_width <= 1+CELL_SIZE_TOLERANCE and 
                1-CELL_SIZE_TOLERANCE <= h/avg_height <= 1+CELL_SIZE_TOLERANCE)]

def remove_duplicate_cells(cells: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    unique_cells = []
    for cell in cells:
        x1,y1,w1,h1 = cell
        if not any(abs(x1-x2) < w1/2 and abs(y1-y2) < h1/2 
                  for x2,y2,w2,h2 in unique_cells):
            unique_cells.append(cell)
    return unique_cells

def sort_cells(cells: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    cells_with_centers = [(x + w/2, y + h/2, (x,y,w,h)) 
                         for x,y,w,h in cells]
    cells_sorted_by_y = sorted(cells_with_centers, key=lambda x: x[1])
    rows = [cells_sorted_by_y[i:i+7] for i in range(0, CELL_COUNT, 7)]
    return [cell[2] for row in rows for cell in sorted(row, key=lambda x: x[0])]

def extract_grid(image_path: str) -> List[np.ndarray]:
    """
    Extract cells from a grid image
    Args:
        image_path: Path to the input image
    Returns:
        List of numpy arrays containing individual cell images
    """
    img = cv2.imread(image_path)
    thresh = preprocess_image(img)
    grid = detect_grid_lines(thresh)
    
    contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cells = [(x,y,w,h) for cnt in contours 
             if cv2.contourArea(cnt) > MIN_CELL_AREA 
             for x,y,w,h in [cv2.boundingRect(cnt)]]
    
    filtered_cells = filter_cells(cells)
    unique_cells = remove_duplicate_cells(filtered_cells)
    
    if len(unique_cells) != CELL_COUNT:
        raise ValueError(f"Expected {CELL_COUNT} cells, but found {len(unique_cells)}")
    
    sorted_cells = sort_cells(unique_cells)
    
    cell_images = []
    for i, (x,y,w,h) in enumerate(sorted_cells):
        margin_w = int(w * CELL_MARGIN_PERCENT)
        margin_h = int(h * CELL_MARGIN_PERCENT)
        cell = img[y+margin_h:y+h-margin_h, x+margin_w:x+w-margin_w]
        cell_images.append(cell)
        
        # Debug visualization
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    # Debug: Display the detected grid with rectangles
    cv2.imshow('Grid Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cell_images

# Usage example
grid_cells = extract_grid('./sample.jpg')

# Create cell directory if not exists
os.makedirs('cell', exist_ok=True)

# Save extracted cells as individual images
for i, cell in enumerate(grid_cells):
    cv2.imwrite(os.path.join('cell', f'cell_{i}.jpg'), cell)