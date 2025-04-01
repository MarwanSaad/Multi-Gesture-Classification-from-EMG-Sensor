# Classification Notebook

## Overview
This repository contains a Jupyter Notebook for performing classification tasks on Electromyography (EMG) data. The notebook loads and processes multiple datasets representing different movement gestures, applies machine learning classifiers, and evaluates their performance.

## Features
- Loads and preprocesses EMG data from multiple CSV files.
- Labels data with corresponding movement types.
- Implements machine learning classification using:
  - Random Forest
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
- Performs train-test split and cross-validation.
- Computes and visualizes classification metrics.

## Installation
### Prerequisites
Ensure you have Python installed along with Jupyter Notebook. The following libraries are required:
- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`
- `sklearn`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/classification-notebook.git
   cd classification-notebook
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage
1. Open `Classification.ipynb` in Jupyter Notebook.
2. Run the notebook cells step by step:
   - Load datasets
   - Assign movement labels
   - Train machine learning models
   - Evaluate classification results
3. Modify or extend the notebook for additional analysis.

## Data Description
The dataset consists of EMG signals recorded for various hand movements. Each CSV file represents a different movement and contains:
- `timestamp`: Time of signal capture
- `emg1` to `emg8`: EMG channel readings
- `movement`: Assigned movement label (e.g., TAP, victory, open, closed, neutral, flexion, extension)

## Results
- Confusion matrices and classification reports provide insights into model performance.
- Accuracy scores and cross-validation help in selecting the best model.
- Data visualizations illustrate patterns in EMG signals.

## Contributing
Feel free to fork the repository and submit pull requests for improvements or additional functionality.

## Contact
For questions or collaborations, contact ma.saadabbas@gmail.com or open an issue on GitHub.
