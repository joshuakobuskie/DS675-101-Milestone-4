import pandas
from preprocess import preprocess_dataset
from generate_exploratory_metrics import generate_exploratory_metrics
dataframe = pandas.read_csv("../data/Health_Sleep_Statistics.csv")
preprocess_dataset(dataframe)
generate_exploratory_metrics(dataframe)

print(dataframe)
