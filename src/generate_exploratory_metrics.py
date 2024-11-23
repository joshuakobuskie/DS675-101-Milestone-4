import numpy
from matplotlib import pyplot

def generate_exploratory_metrics(dataframe):
	correlation_matrix = dataframe.corr()
	feature_names = dataframe.columns.to_numpy()
	
	fig, ax = pyplot.subplots(figsize=(6, 5))
	pyplot.title('Correlation matrix')
	pyplot.xticks(rotation=90, ticks=range(0, 11), labels = feature_names)
	pyplot.yticks(ticks=range(0, 11), labels = feature_names)
	pyplot.imshow(correlation_matrix)
	pyplot.tight_layout()
	
	for i in range(11):
		for j in range(11):
			value = correlation_matrix[feature_names[i]][feature_names[j]]
			ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=7, color = 'black' if value > 0 else 'white')

	pyplot.savefig('../assets/correlation_matrix.jpg')
