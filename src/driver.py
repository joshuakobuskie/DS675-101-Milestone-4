import pandas
from preprocess import preprocess_dataset
from generate_exploratory_metrics import generate_exploratory_metrics
#from split_data import train_test_split

dataframe = pandas.read_csv("../data/Health_Sleep_Statistics.csv")
# Split train/test first
# Preprocessing should be called twice, once of the train data and once on the test data
# This ensures that we dont have data leakage
preprocess_dataset(dataframe)

from sklearn.model_selection import train_test_split
train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.1, random_state=42, stratify=dataframe["Sleep Quality"])
#train_data, test_data = train_test_split(dataframe, dataframe["Sleep Quality"])

from sklearn.preprocessing import StandardScaler

# Select columns to scale and standard scale based on training data only
scaling_columns = ["Age", "Bedtime", "Wake-up Time", "Daily Steps", "Calories Burned"]
scaler = StandardScaler()
train_dataframe[scaling_columns] = scaler.fit_transform(train_dataframe[scaling_columns])
test_dataframe[scaling_columns] = scaler.transform(test_dataframe[scaling_columns])

generate_exploratory_metrics(dataframe)
print(dataframe)
print(train_dataframe)
print(test_dataframe)
