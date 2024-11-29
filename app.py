import streamlit as st
import os
import numpy as np
import cv2
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
        st.write("Failed to solve the Nurikabe puzzle")
        return None
    
    # Print solution to console
    st.write("Nurikabe Puzzle Solution:")
    st.table(solution)
    
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

def display_grid(grid, title):
    st.write(title)
    st.table(grid)

def display_cells(cell_dir, title):
    st.write(title)
    cell_images = []
    for i in range(49):
        cell_path = os.path.join(cell_dir, f'cell_{i}.jpg')
        cell_image = cv2.imread(cell_path)
        cell_images.append(cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB))
    
    # Create a 7x7 grid layout
    for row in range(7):
        cols = st.columns(7)
        for col in range(7):
            idx = row * 7 + col
            with cols[col]:
                st.image(cell_images[idx], caption=f'Cell {idx}', use_container_width=True)

def main():
    st.title("Nurikabe Puzzle Solver")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded image
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the original image
        original_image = cv2.imread('uploaded_image.jpg')
        st.image(original_image, channels="BGR", caption="Original Image")
        
        # Step 1: Extract grid cells
        extract_grid('uploaded_image.jpg')
        
        # Display extracted cells in a 7x7 table format
        display_cells('cell', "Extracted Cells")
        
        # Step 2: Predict numbers in cells
        input_data = read_grid_from_prediction()
        
        # Read and display the grid from grid_output.txt
        with open('grid_output.txt', 'r') as f:
            lines = f.readlines()
        grid = [line.strip().split() for line in lines]
        display_grid(grid, "Predicted Grid")
        
        # Step 3: Solve Nurikabe puzzle
        solution = solve_and_visualize_nurikabe(input_data)
        
        if solution is not None:
            # Step 4: Color walls black
            color_walls_black(solution, 'uploaded_image.jpg')
            st.write("Nurikabe puzzle solved and walls colored black successfully!")
            
            # Display the final result image
            final_image = cv2.imread('uploaded_image_black.jpg')
            st.image(final_image, channels="BGR", caption="Final Result Image")

if __name__ == "__main__":
    main()
