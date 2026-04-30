# type: ignore
import kagglehub
import pandas as pd

#Download a model from kaggle to start our ML work
path = kagglehub.dataset_download("nalisha/job-salary-prediction-dataset")
print("Path to dataset files:", path)


df_train = pd.read_csv(path + "/job_salary_prediction_dataset.csv")

#take 100 of data for test reasons
df_test = df_train.sample(1000, random_state=42)
df_train = df_train.drop(df_test.index)

input_features_x = df_train.iloc[:,:-1].values
output_targets_y = df_train.iloc[:,-1:].values.flatten() #.flatten to have a ndarray of shape (m,) instead of (m,1)


