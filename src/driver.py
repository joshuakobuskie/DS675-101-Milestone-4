import pandas
from preprocess import preprocess_dataset
from generate_exploratory_metrics import generate_exploratory_metrics
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from metrics import get_perf_metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shap
#from split_data import train_test_split

dataframe = pandas.read_csv("../data/Health_Sleep_Statistics.csv")
# Split train/test first
# Preprocessing should be called twice, once of the train data and once on the test data
# This ensures that we dont have data leakage
preprocess_dataset(dataframe)

y_data = dataframe["Sleep Quality"]
x_data = dataframe.drop(["Sleep Quality"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42, stratify=dataframe["Sleep Quality"])

# Select columns to scale and standard scale based on training data only
scaling_columns = ["Age", "Bedtime", "Wake-up Time", "Daily Steps", "Calories Burned"]
scaler = StandardScaler()
x_train[scaling_columns] = scaler.fit_transform(x_train[scaling_columns])
x_test[scaling_columns] = scaler.transform(x_test[scaling_columns])

generate_exploratory_metrics(dataframe)

# Random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100) 
random_forest.fit(x_train, y_train)
random_forest_pred = random_forest.predict(x_test)
random_forest_proba = random_forest.predict_proba(x_test)

#Metrics
precision, recall, f1, accuracy, auc = get_perf_metrics(y_test, random_forest_pred, random_forest_proba)

#This could definitely just be defined as a function and called
#Confusion Matrix
conf_matrix = confusion_matrix(y_test, random_forest_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g')
pyplot.title("Confusion Matrix")
pyplot.xlabel("Predicted Label")
pyplot.ylabel("True Label")
pyplot.tight_layout()
pyplot.savefig('../assets/random_forest_confusion_matrix.jpg')
pyplot.clf()
pyplot.close()

#This could definitely just be defined as a function and called
#SHAP values
explainer = shap.Explainer(random_forest.predict, x_test)
shap_values = explainer(x_test)

shap.plots.bar(shap_values, show=False)
pyplot.savefig('../assets/random_forest_shap_values_bar.jpg')
pyplot.clf()
pyplot.close()

shap.summary_plot(shap_values, show=False)
pyplot.savefig('../assets/random_forest_shap_values_summary.jpg')
pyplot.clf()
pyplot.close()

shap.summary_plot(shap_values, plot_type='violin', show=False)
pyplot.savefig('../assets/random_forest_shap_values_violin.jpg')
pyplot.clf()
pyplot.close()

print("Random Forest Classifier")
print("Precision: {}, Recall: {}, F1: {}, Accuracy: {}, AUC: {}".format(precision, recall, f1, accuracy, auc))

# Support Vector classifier
svm = SVC(probability=True) 
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
svm_proba = svm.predict_proba(x_test)

#Metrics
precision, recall, f1, accuracy, auc = get_perf_metrics(y_test, svm_pred, svm_proba)

#This could definitely just be defined as a function and called
#Confusion Matrix
conf_matrix = confusion_matrix(y_test, random_forest_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g')
pyplot.title("Confusion Matrix")
pyplot.xlabel("Predicted Label")
pyplot.ylabel("True Label")
pyplot.tight_layout()
pyplot.savefig('../assets/svm_confusion_matrix.jpg')
pyplot.clf()
pyplot.close()

#This could definitely just be defined as a function and called
#SHAP Values
explainer = shap.Explainer(svm.predict, x_test)
shap_values = explainer(x_test)

shap.plots.bar(shap_values, show=False)
pyplot.savefig('../assets/svm_shap_values_bar.jpg')
pyplot.clf()
pyplot.close()

shap.summary_plot(shap_values, show=False)
pyplot.savefig('../assets/svm_shap_values_summary.jpg')
pyplot.clf()
pyplot.close()
shap.summary_plot(shap_values, plot_type='violin', show=False)
pyplot.savefig('../assets/svm_shap_values_violin.jpg')
pyplot.clf()
pyplot.close()

print("SVM")
print("Precision: {}, Recall: {}, F1: {}, Accuracy: {}, AUC: {}".format(precision, recall, f1, accuracy, auc))
