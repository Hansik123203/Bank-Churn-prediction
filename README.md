# Bank-Churn-prediction
### Kaggle dataset link: https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling/code
This project focuses on predicting customer churn for a bank using machine learning techniques. The objective is to build a model that can predict whether a customer will leave (churn) or stay with the bank based on their demographic and financial details. The model is built using a feed-forward neural network with data preprocessing, feature engineering, and model evaluation as key components of the workflow.
## Dataset Overview
The dataset used in this project is the Bank Churn Modelling Dataset, which contains customer information along with a target variable indicating whether the customer has churned. Key features include:
•	Age
•	Tenure (length of time the customer has been with the bank)
•	Credit Score
•	Balance
•	Estimated Salary
•	NumOfProducts (number of products the customer has with the bank)
•	Geography (country of residence)
•	Gender
The target variable, Exited, indicates whether a customer has churned (1) or stayed (0).
## Project Overview
### 1.	Data Preprocessing:
o	Removed irrelevant columns like RowNumber, CustomerId, and Surname to focus on the features that impact churn.
o	Checked for missing data and handled it accordingly.
o	Performed one-hot encoding for categorical variables (Geography and Gender) to convert them into numerical representations suitable for model training.
o	Applied Min-Max scaling to normalize continuous features for better performance in the neural network.

### 2.	Exploratory Data Analysis (EDA):
o	Visualized the distribution of key features like CreditScore, Age, Tenure, and EstimatedSalary using histograms.
o	Plotted comparisons of these features based on customer churn status to gain insights into the characteristics of churned vs. retained customers.

### 3.	Modeling:
o	Built an artificial neural network (ANN) using TensorFlow/Keras with:
	Input layer (13 features)
	Two hidden layers with ReLU activation
	Output layer with a sigmoid activation for binary classification
o	The model was trained on the processed data for 50 epochs.

### 4.	Model Evaluation:
o	Evaluated the model's performance using a test set and calculated metrics such as accuracy, precision, and recall.
o	Generated a confusion matrix to visualize the model's classification results.
o	The model’s ability to correctly predict both churned and retained customers was assessed.

## Technologies Used
•	Python: Core programming language
•	Pandas: Data manipulation and preprocessing
•	NumPy: Numerical operations
•	Matplotlib & Seaborn: Data visualization
•	Scikit-learn: Data preprocessing and model evaluation
•	TensorFlow/Keras: Building and training the artificial neural network (ANN)
•	
## How to Run the Project
1.	Clone the repository using:
bash
CopyEdit
git clone https://github.com/your-username/bank-churn-prediction.git
2.	Install the required dependencies:
bash
CopyEdit
pip install -r requirements.txt
3.	Run the Jupyter notebook (Bank_turnover_churn.ipynb) to execute the steps outlined in the project. The notebook contains the entire workflow, from data loading to model evaluation.

## Future Enhancements
•	Hyperparameter Tuning: Perform grid search or random search to optimize model parameters.
•	Model Comparison: Evaluate other machine learning models (e.g., Random Forest, Support Vector Machine) to compare with the neural network's performance.
•	Feature Engineering: Experiment with additional features or transformations to improve model accuracy.
•	Model Deployment: Deploy the model using a web framework (e.g., Flask or FastAPI) to create an interactive churn prediction tool.
