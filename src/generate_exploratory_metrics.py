import numpy
from matplotlib import pyplot

def generate_exploratory_metrics(dataframe):
	correlation_matrix = dataframe.corr()
	feature_names = dataframe.columns.to_numpy()
	
	pyplot.figure()
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

	pyplot.figure()
	fig, ax = pyplot.subplots(4, 3, figsize=(12, 10))
	for i in range(11):
		column = dataframe[feature_names[i]]
		ax[i // 3, i % 3].hist(column)
		ax[i // 3, i % 3].set_title(feature_names[i] + " distribution")

	pyplot.subplots_adjust(hspace=0.5)
	pyplot.savefig('../assets/distribution_plots.jpg')

	pyplot.figure()
	fig, ax = pyplot.subplots(figsize=(10, 3))
	pyplot.subplots_adjust(hspace=0)
	pyplot.axis('off')
	pyplot.title("Correlation between each variable and Sleep Quality")
	table = pyplot.table(cellText=[[f'{value:.2f}' for value in correlation_matrix['Sleep Quality'].to_numpy()]], colLabels=[name.replace(' ', '\n') for name in feature_names])
	table.scale(1, 3)
	table.auto_set_font_size(False)
	table.set_fontsize(9)
	pyplot.savefig('../assets/target_variable_correlations.jpg', bbox_inches='tight')
