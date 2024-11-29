# Hand-written Nurikabe Puzzle Solver 🧩

## Team Information
**Team 20 - Visionary 4**
Course: Image Processing
Institution: Chung-ang University

## Project Overview
The Hand-written Nurikabe Puzzle Solver is an advanced computer vision and AI-powered application designed to recognize hand-drawn Nurikabe puzzles, solve them, and visualize the solution directly on the original image.

## Project Description
Nurikabe is a logic puzzle where the goal is to color cells black or white according to specific rules:
- White cells form "islands" with a number indicating the island's size
- Black cells form a continuous "sea"
- No 2x2 block of black cells is allowed
- Islands cannot touch each other

Our project uses a sophisticated machine learning approach to:
1. Extract grid cells from an input image
2. Recognize digits in each cell
3. Solve the Nurikabe puzzle using advanced logical deduction
4. Visualize the solution by coloring black cells on the original image

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (strongly recommended)

### Setup
1. Clone the repository
```bash
git clone https://github.com/mldljyh/Image_Processing_Project.git
cd Image_Processing_Project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download Pre-trained Model
Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1WghpuaFl15KYH0gcjTEim6GPDh-flhVJ/view?usp=drive_link) and place it in the `checkpoints` directory.

## Usage

### Running the Application
```bash
streamlit run app.py
```

### How to Use
1. Upload a hand-drawn Nurikabe puzzle image
2. The application will:
   - Extract individual cell images
   - Predict digits in each cell
   - Solve the Nurikabe puzzle
   - Display the solution with black cells colored

## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy
- Streamlit
- EnsNet

## Project Structure
```
nurikabe-puzzle-solver/
│
├── checkpoints/             # Trained model weights
├── app.py                   # Streamlit web application
├── predict.py               # Digit recognition model
├── nurikabe_solver.py       # Nurikabe puzzle solving logic
├── extract.py               # Grid and cell extraction
├── fill_black.py            # Image manipulation
├── EnsNet.py                # Ensemble neural network model
└── requirements.txt         # Project dependencies
```

## Model Architecture
The project uses an advanced ensemble neural network (EnsNet) with:
- Base CNN for feature extraction
- Multiple fully connected subnets
- Ensemble prediction for robust digit recognition

## Performance
- Supports 7x7 grid Nurikabe puzzles
- Handles hand-drawn input
- Ensemble model provides high accuracy in digit recognition