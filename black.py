import cv2
import numpy as np
from extract import extract_grid, preprocess_image, detect_grid_lines, filter_cells, remove_duplicate_cells, sort_cells

def connected_components(binary_image: np.ndarray) -> np.ndarray:
    """
    Detect connected components using NumPy operations
    Returns labeled array of connected components
    """
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
    return labeled

def detect_cells_numpy(grid: np.ndarray) -> list:
    """
    Detect cells using NumPy connected components
    """
    labeled = connected_components(grid)
    cells = []
    
    for label_val in np.unique(labeled)[1:]:  # Skip background (0)
        component = (labeled == label_val)
        if np.sum(component) > 1000:  # Equivalent to contourArea check
            ys, xs = np.where(component)
            x, y = xs.min(), ys.min()
            w, h = xs.max() - x + 1, ys.max() - y + 1
            cells.append((x, y, w, h))
    
    return cells

def fill_cell_black(image_path: str, cell_indexes, debug: bool = False) -> str:
    """
    Fill a specific cell with black color and save as a new image
    Args:
        image_path: Path to the original image
        cell_index: Index of the cell to fill (0-based, default 24 is the 25th cell)
        debug: Whether to show debug visualizations
    Returns:
        Path to the output image
    """
    # Read original image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert to NumPy for processing
    img_np = np.array(img)
    
    # Preprocess using NumPy methods
    thresh = preprocess_image(img_np)
    grid = detect_grid_lines(thresh)
    
    # Detect cells using NumPy method
    cells = detect_cells_numpy(grid)
    
    filtered_cells = filter_cells(cells)
    unique_cells = remove_duplicate_cells(filtered_cells)
    sorted_cells = sort_cells(unique_cells)
    
    for cell_index in cell_indexes:
    # Fill the specified cell with black
        if 0 <= cell_index < len(sorted_cells):
            
            x, y, w, h = sorted_cells[cell_index]
            img[y:y+h, x:x+w] = [0, 0, 0]  # OpenCV-style rectangle fill
    
    # Save result using OpenCV
    output_path = image_path.rsplit('.', 1)[0] + '_black.jpg'
    cv2.imwrite(output_path, img)
    print(f"Image has been saved: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Usage example
    fill_cell_black('./sample.jpg')