import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np

def sort_dataset(dataset_df):
    sorted_df = dataset_df.sort_values(by='year')
    return sorted_df

def split_dataset(dataset_df):
   
    dataset_df['salary'] *= 0.001

    train_df = dataset_df.loc[:1717]  # 인덱스 범위 [:1718]
    test_df = dataset_df.loc[1718:]  # 인덱스 범위 [1718:]

    X_train, X_test, Y_train, Y_test = train_test_split(train_df.drop('salary', axis=1), train_df['salary'], test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test


def extract_numerical_cols(dataset_df):
    numerical_columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    numerical_df = dataset_df[numerical_columns]

    return numerical_df

def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_regressor = DecisionTreeRegressor(random_state=42)

    dt_regressor.fit(X_train, Y_train)

    predictions = dt_regressor.predict(X_test)

    return predictions

def train_predict_random_forest(X_train, Y_train, X_test):
    rf_regressor = RandomForestRegressor(random_state=42)

    rf_regressor.fit(X_train, Y_train)

    predictions = rf_regressor.predict(X_test)

    return predictions

def train_predict_svm(X_train, Y_train, X_test) :
    svm_pipeline = make_pipeline(StandardScaler(), SVR())

    svm_pipeline.fit(X_train, Y_train)

    predictions = svm_pipeline.predict(X_test)

    return predictions

def calculate_RMSE(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)

    squared_errors = (labels - predictions) ** 2

    mean_squared_error = np.mean(squared_errors)

    rmse = np.sqrt(mean_squared_error)

    return rmse


if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

	sorted_df = sort_dataset(data_df)
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)

	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
