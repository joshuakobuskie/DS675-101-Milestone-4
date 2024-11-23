import pandas

def preprocess_dataset(dataframe):
	#Drop the User ID (Not a useful feature)
	dataframe.drop(['User ID'], axis=1, inplace=True)

	# Convert categorical data to numerical data

	# Ordinal data
	dataframe["Physical Activity Level"] = dataframe["Physical Activity Level"].map({
		'low' : -1,
		'medium' : 0,
		'high': 1
	})

	dataframe["Dietary Habits"] = dataframe["Dietary Habits"].map({
		'healthy' : 1,
		'medium' : 0,
		'unhealthy' : -1,
	})

	# Binary data
	dataframe["Sleep Disorders"] = dataframe["Sleep Disorders"].map({
		'no' : -1,
		'yes' : 1,
	})

	dataframe["Medication Usage"] = dataframe["Medication Usage"].map({
		'no' : -1,
		'yes' : 1,
	})
	
	dataframe["Gender"] = dataframe["Gender"].map({
		'm' : -1,
		'f': 1
	})

	#Convert timestamps to numerical
	bedtimes = pandas.to_datetime(dataframe['Bedtime'], format='%H:%M')
	dataframe['Bedtime'] = bedtimes.dt.hour * 60 + bedtimes.dt.minute

	wake_times = pandas.to_datetime(dataframe['Wake-up Time'], format='%H:%M')
	dataframe['Wake-up Time'] = wake_times.dt.hour * 60 + wake_times.dt.minute

	#Standard normalization skipping ordinal data and target variable
	for column in dataframe.columns:
		if not column in ["Gender", "Physical Activity Level", "Dietary Habits", "Sleep Disorders", "Medication Usage", "Sleep Quality"]:
			dataframe[column] -= dataframe[column].mean()
			dataframe[column] /= dataframe[column].std()

