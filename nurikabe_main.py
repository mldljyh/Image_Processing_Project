import os
import numpy as np
import cv2

# Import functions from existing modules
from extract import extract_grid
from predict import main as predict_numbers
from nurikabe_solver import solve_nurikabe_puzzle, State
from fill_black import fill_cell_black

def read_grid_from_prediction():
    """
    Read the grid of numbers from the prediction results
    Returns a list of tuples (number, x, y) for Nurikabe solver
    """
    # Ensure grid_output.txt exists
    if not os.path.exists('grid_output.txt'):
        predict_numbers()
    
    # Read the grid from the output file
    with open('grid_output.txt', 'r') as f:
        lines = f.readlines()
    
    input_data = []
    for y, line in enumerate(lines):
        row = line.strip().split()
        for x, val in enumerate(row):
            if val != 'X' and val != '':
                input_data.append((int(val), x, y))
    
    return input_data

def solve_and_visualize_nurikabe(input_data):
    """
    Solve the Nurikabe puzzle and visualize the solution
    
    Args:
        input_data: List of tuples (number, x, y)
    Returns:
        Solved puzzle numpy array
    """
    solution, success = solve_nurikabe_puzzle(input_data)
    
    if not success:
        print("Failed to solve the Nurikabe puzzle")
        return None
    
    # Print solution to console
    print("Nurikabe Puzzle Solution:")
    print(solution)
    
    return solution

def color_walls_black(solution, original_image_path='sample.jpg'):
    """
    Color cells corresponding to walls (-1) black
    
    Args:
        solution: Numpy array of solved Nurikabe puzzle
        original_image_path: Path to the original image
    """
    # Find indices of wall cells
    wall_coordinates = np.where(solution == State.SEA)

    cell_indexes = []
    for y, x in list(zip(wall_coordinates[0], wall_coordinates[1])):
        cell_indexes.append(y*7 + x)
    fill_cell_black(original_image_path, cell_indexes)

def main():
    # Step 1: Extract grid cells
    extract_grid('./sample.jpg')
    
    # Step 2: Predict numbers in cells
    input_data = read_grid_from_prediction()
    
    # Step 3: Solve Nurikabe puzzle
    solution = solve_and_visualize_nurikabe(input_data)
    
    if solution is not None:
        # Step 4: Color walls black
        color_walls_black(solution)
        print("Nurikabe puzzle solved and walls colored black successfully!")

if __name__ == "__main__":
    main()