import pandas
from preprocess import preprocess_dataset
from generate_exploratory_metrics import generate_exploratory_metrics
from split_data import train_test_split

dataframe = pandas.read_csv("../data/Health_Sleep_Statistics.csv")
# Split train/test first
# Preprocessing should be called twice, once of the train data and once on the test data
# This ensures that we dont have data leakage
train_data, test_data = train_test_split(dataframe)

preprocess_dataset(dataframe)
generate_exploratory_metrics(dataframe)
print(dataframe)
