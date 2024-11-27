import pandas
from preprocess import preprocess_dataset
from generate_exploratory_metrics import generate_exploratory_metrics
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
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

precision = precision_score(y_test, random_forest_pred, average='macro')
recall = recall_score(y_test, random_forest_pred, average='macro')
f1 = f1_score(y_test, random_forest_pred, average='macro')
accuracy = accuracy_score(y_test, random_forest_pred)
auc = roc_auc_score(y_test, random_forest_proba, multi_class='ovr')

explainer = shap.Explainer(random_forest.predict, x_test)
shap_values = explainer(x_test)

shap.plots.bar(shap_values)
pyplot.savefig('../assets/random_forest_shap_values_bar.jpg')
pyplot.clf()
pyplot.close()

shap.summary_plot(shap_values)
pyplot.savefig('../assets/random_forest_shap_values_summary.jpg')
pyplot.clf()
pyplot.close()

shap.summary_plot(shap_values, plot_type='violin')
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

precision = precision_score(y_test, svm_pred, average='macro')
recall = recall_score(y_test, svm_pred, average='macro')
f1 = f1_score(y_test, svm_pred, average='macro')
accuracy = accuracy_score(y_test, svm_pred)
auc = roc_auc_score(y_test, svm_proba, multi_class='ovr')

explainer = shap.Explainer(svm.predict, x_test)
shap_values = explainer(x_test)

shap.plots.bar(shap_values)
pyplot.savefig('../assets/svm_shap_values_bar.jpg')
pyplot.clf()
pyplot.close()

shap.summary_plot(shap_values)
pyplot.savefig('../assets/svm_shap_values_summary.jpg')
pyplot.clf()
pyplot.close()
shap.summary_plot(shap_values, plot_type='violin')
pyplot.savefig('../assets/svm_shap_values_violin.jpg')
pyplot.clf()
pyplot.close()

print("SVM")
print("Precision: {}, Recall: {}, F1: {}, Accuracy: {}, AUC: {}".format(precision, recall, f1, accuracy, auc))
