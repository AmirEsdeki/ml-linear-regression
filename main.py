import kagglehub
import pandas as pd

#Download a model from kaggle to start our ML work
path = kagglehub.dataset_download("nalisha/job-salary-prediction-dataset")
print("Path to dataset files:", path)


data_csv = pd.read_csv(path + "/job_salary_prediction_dataset.csv")
print(data_csv.head(10))
# features = data_csv.iloc[]