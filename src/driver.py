import pandas
from preprocess import preprocess_dataset

dataframe = pandas.read_csv("../data/Health_Sleep_Statistics.csv")
preprocess_dataset(dataframe)
print(dataframe)