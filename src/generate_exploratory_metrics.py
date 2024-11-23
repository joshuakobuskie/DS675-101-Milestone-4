import numpy
from matplotlib import pyplot

def generate_exploratory_metrics(dataframe):
	correlation_matrix = dataframe.corr()
	feature_names = dataframe.columns.to_numpy()
	
	fig, ax = pyplot.subplots(figsize=(10, 10))
	pyplot.title('Correlation matrix')
	pyplot.xticks(rotation=90, ticks=range(len(feature_names)), labels = feature_names, fontsize=15)
	pyplot.yticks(ticks=range(len(feature_names)), labels = feature_names, fontsize=15)
	pyplot.imshow(correlation_matrix)
	pyplot.tight_layout()
	
	for i in range(len(feature_names)):
		for j in range(len(feature_names)):
			value = correlation_matrix[feature_names[i]][feature_names[j]]
			ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=15, color = 'black' if value > 0 else 'white')

	pyplot.savefig('../assets/correlation_matrix.jpg')
