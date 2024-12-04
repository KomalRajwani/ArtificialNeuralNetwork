"Artificial Neural Network implementation using drugdataset.csv for classification."
# Neural Network Implementation - DATA1200

## Project Title
Artificial Neural Network for Drug Classification

## Short Description
This project involves building and evaluating an Artificial Neural Network (ANN) using Python for the classification of drugs based on patient attributes such as age, sex, blood pressure, cholesterol, and sodium-to-potassium ratio. A Logistic Regression model is also implemented for comparison.

## Getting Started
This section explains how to set up and run the project.

### Prerequisites
- Python 3.8 or higher
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

### Installing
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/<KomalRajwani>/NeuralNetwork-DATA1200.git
### Navigate to the project directory:
bash
cd NeuralNetwork-
## Install the required Python libraries:
bash

pip install pandas numpy matplotlib scikit-learn
Running the Tests
Place the drugdataset.csv file in the project directory.
## Run the script:
bash
python ann_model.py
## Breakdown of Tests
Data Preparation:
Categorical variables (e.g., sex, blood pressure) are encoded.
Features are standardized using StandardScaler to improve model performance.
Model Training:
An Artificial Neural Network (ANN) with (5, 4, 5) hidden layers is trained using the MLPClassifier.
A Logistic Regression model is trained as a baseline for comparison.
Model Evaluation:
Confusion matrices and classification reports are generated for both models to evaluate precision, recall, F1-score, and overall accuracy.
## Deployment
This project is designed for educational purposes. For deployment in a production environment:

Host the trained model on a cloud platform like AWS, Azure, or Google Cloud.
Build a web or mobile application to interface with the model.

## Author
Komal Rajwani
