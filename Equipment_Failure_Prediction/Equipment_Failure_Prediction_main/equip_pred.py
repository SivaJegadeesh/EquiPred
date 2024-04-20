import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("iot_sensor_dataset.csv")

# Extract features and target variable
X = data.iloc[:, :-1]  # Selecting all columns except the last one as features
y = data.iloc[:, -1]   # Selecting the last column as the target variable

# Convert to numpy arrays and handle data type conversion if necessary
X = X.values
y = y.values.astype('int')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the logistic regression model
knn_model = KNeighborsClassifier(n_neighbors=5) 

# Train the model using all features
knn_model.fit(X_train, y_train)

# Save the trained model to a file
pickle.dump(knn_model, open('model.pkl', 'wb'))

# Load the model from the file (optional)
# model = pickle.load(open('model.pkl', 'rb'))
