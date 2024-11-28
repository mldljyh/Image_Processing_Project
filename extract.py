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

def create_gaussian_kernel(size=5, sigma=1):
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
    return kernel / kernel.sum()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale using numpy
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    
    # Apply gaussian blur using numpy convolution
    kernel = create_gaussian_kernel(5)
    blur = convolve2d(gray, kernel)
    
    # Adaptive thresholding using numpy operations
    local_mean = uniform_filter(blur, size=11)
    threshold = local_mean - 2
    return (blur < threshold).astype(np.uint8) * 255

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Implement 2D convolution using numpy's stride_tricks
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    shape = (image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1])
    strides = padded_image.strides * 2
    sub_matrices = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)
    output = np.einsum('ijkl,kl->ij', sub_matrices, kernel)
    return output

def uniform_filter(image: np.ndarray, size: int) -> np.ndarray:
    # Implement uniform filter using numpy's stride_tricks
    kernel = np.ones((size, size)) / (size * size)
    return convolve2d(image, kernel)

def detect_grid_lines(thresh: np.ndarray) -> np.ndarray:
    horizontal = thresh.copy()
    vertical = thresh.copy()
    
    # Horizontal lines
    cols = horizontal.shape[1]
    horizontal_size = cols // GRID_DIVISION_FACTOR
    kernel = np.ones((1, horizontal_size))
    horizontal = minimum_filter(horizontal, (1, horizontal_size))
    horizontal = maximum_filter(horizontal, (1, horizontal_size))
    
    # Vertical lines
    rows = vertical.shape[0]
    vertical_size = rows // GRID_DIVISION_FACTOR
    kernel = np.ones((vertical_size, 1))
    vertical = minimum_filter(vertical, (vertical_size, 1))
    vertical = maximum_filter(vertical, (vertical_size, 1))
    
    return np.maximum(horizontal, vertical)

def minimum_filter(image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    # Implement minimum filter using numpy's stride_tricks
    pad_h, pad_w = kernel_size[0] // 2, kernel_size[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    shape = (image.shape[0], image.shape[1], kernel_size[0], kernel_size[1])
    strides = padded_image.strides * 2
    sub_matrices = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)
    output = np.min(sub_matrices, axis=(2,3))
    return output

def maximum_filter(image: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    # Implement maximum filter using numpy's stride_tricks
    pad_h, pad_w = kernel_size[0] // 2, kernel_size[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    shape = (image.shape[0], image.shape[1], kernel_size[0], kernel_size[1])
    strides = padded_image.strides * 2
    sub_matrices = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)
    output = np.max(sub_matrices, axis=(2,3))
    return output

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

def detect_cells(grid: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Use numpy operations to detect connected components
    labeled, num_features = connected_components(grid)
    cells = []
    
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if np.sum(component) > MIN_CELL_AREA:
            ys, xs = np.where(component)
            x, y = xs.min(), ys.min()
            w, h = xs.max() - x + 1, ys.max() - y + 1
            cells.append((x, y, w, h))
    
    return cells

def connected_components(binary_image: np.ndarray) -> Tuple[np.ndarray, int]:
    labeled = np.zeros_like(binary_image, dtype=int)
    label = 1
    stack = []
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] and not labeled[i, j]:
                stack.append((i, j))
                while stack:
                    x, y = stack.pop()
                    if (0 <= x < binary_image.shape[0]) and (0 <= y < binary_image.shape[1]):
                        if binary_image[x, y] and not labeled[x, y]:
                            labeled[x, y] = label
                            stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
                label += 1
    return labeled, label - 1

def extract_grid(image_path: str, debug: bool = False) -> List[np.ndarray]:
    """
    Extract cells from a grid image
    Args:
        image_path: Path to the input image
        debug: Whether to show debug visualizations
    Returns:
        List of numpy arrays containing individual cell images
    """
    img = cv2.imread(image_path)
    
    # Preprocess and detect grid
    thresh = preprocess_image(img)
    grid = detect_grid_lines(thresh)
    
    cells = detect_cells(grid)
    filtered_cells = filter_cells(cells)
    unique_cells = remove_duplicate_cells(filtered_cells)
    
    # Warn if cell count is incorrect
    if len(unique_cells) != CELL_COUNT:
        print(f"Warning: Expected {CELL_COUNT} cells, but found {len(unique_cells)}")
    
    sorted_cells = sort_cells(unique_cells)
    
    # Create cell directory if not exists
    os.makedirs('cell', exist_ok=True)
    
    # Save and return extracted cells
    extracted_cells = []
    for i, (x, y, w, h) in enumerate(sorted_cells):
        margin_w = int(w * CELL_MARGIN_PERCENT)
        margin_h = int(h * CELL_MARGIN_PERCENT)
        cell = img[y+margin_h:y+h-margin_h, x+margin_w:x+w-margin_w]
        
        # Save cell image
        cell_path = os.path.join('cell', f'cell_{i}.jpg')
        cv2.imwrite(cell_path, cell)
        
        extracted_cells.append(cell)
    
    return extracted_cells

# Usage example
if __name__ == "__main__":
    extract_grid('./sample.jpg')