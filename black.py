import cv2
import numpy as np
from extract import extract_grid, preprocess_image, detect_grid_lines, filter_cells, remove_duplicate_cells, sort_cells

def fill_cell_black(image_path: str, cell_index: int = 24) -> None:
    """
    Fill a specific cell with black color and save as a new image
    Args:
        image_path: Path to the original image
        cell_index: Index of the cell to fill (0-based, default 24 is the 25th cell)
    """
    # Read original image
    img = cv2.imread(image_path)
    thresh = preprocess_image(img)
    grid = detect_grid_lines(thresh)
    
    # Detect cells
    contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cells = [(x,y,w,h) for cnt in contours 
             if cv2.contourArea(cnt) > 1000 
             for x,y,w,h in [cv2.boundingRect(cnt)]]
    
    filtered_cells = filter_cells(cells)
    unique_cells = remove_duplicate_cells(filtered_cells)
    sorted_cells = sort_cells(unique_cells)
    
    # Fill the specified cell with black
    if 0 <= cell_index < len(sorted_cells):
        x, y, w, h = sorted_cells[cell_index]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)  # -1 is the option to fill the interior
    
    # Save result
    output_path = image_path.rsplit('.', 1)[0] + '_black.jpg'
    cv2.imwrite(output_path, img)
    print(f"Image has been saved: {output_path}")

if __name__ == "__main__":
    # Usage example
    fill_cell_black('./sample.jpg')
