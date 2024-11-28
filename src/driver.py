import pandas
from preprocess import preprocess_dataset
from generate_exploratory_metrics import generate_exploratory_metrics
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from metrics import get_perf_metrics, save_conf_matrix, save_shap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shap
import optuna
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


n_estimators_values = []
random_forest_f1_scores = []
# Random forest classifier
def random_forest_optimizer(trial):
	n_estimators = trial.suggest_int('n_estimators', 1, 20)
	random_forest = RandomForestClassifier(n_estimators = n_estimators) 
	random_forest.fit(x_train, y_train)
	random_forest_pred = random_forest.predict(x_test)
	random_forest_proba = random_forest.predict_proba(x_test)

	#Metrics
	precision, recall, f1, accuracy, auc = get_perf_metrics(y_test, random_forest_pred, random_forest_proba)
	n_estimators_values.append(n_estimators)
	random_forest_f1_scores.append(f1)
	return f1

study = optuna.create_study(direction="maximize")
study.optimize(random_forest_optimizer, n_trials=50)
n_estimators_optimal = study.best_params['n_estimators']

#Hyperparameter optimization graph
pyplot.figure()
pyplot.title("n_estimators vs f1 score")
pyplot.xlabel("n_estimators")
pyplot.ylabel("F1 score")
pyplot.scatter(n_estimators_values, random_forest_f1_scores)
pyplot.savefig("../assets/random_forest_hyperparameter_optimization.jpg")

random_forest = RandomForestClassifier(n_estimators = n_estimators_optimal) 
random_forest.fit(x_train, y_train)
random_forest_pred = random_forest.predict(x_test)
random_forest_proba = random_forest.predict_proba(x_test)

#Metrics
precision, recall, f1, accuracy, auc = get_perf_metrics(y_test, random_forest_pred, random_forest_proba)
print("Optimized Random Forest Classifier")
print("Precision: {}, Recall: {}, F1: {}, Accuracy: {}, AUC: {}".format(precision, recall, f1, accuracy, auc))

#Confusion Matrix
save_conf_matrix(y_test, random_forest_pred, '../assets/random_forest_confusion_matrix.jpg')

#SHAP values
save_shap(random_forest.predict, x_test, "random_forest")

# Support Vector classifier
gamma_values = []
svm_f1_scores = []
def svm_optimizer(trial):
	gamma = trial.suggest_float('gamma', 0.0001, 0.3)
	gamma_values.append(gamma)
	svm = SVC(probability=True, kernel="poly", gamma=gamma) 
	svm.fit(x_train, y_train)
	svm_pred = svm.predict(x_test)
	svm_proba = svm.predict_proba(x_test)

	#Metrics
	precision, recall, f1, accuracy, auc = get_perf_metrics(y_test, svm_pred, svm_proba)

	svm_f1_scores.append(f1)
	return f1

study = optuna.create_study(direction="maximize")
study.optimize(svm_optimizer, n_trials=50)
gamma_optimal = study.best_params['gamma']

svm = SVC(probability=True, gamma=gamma_optimal) 
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
svm_proba = svm.predict_proba(x_test)

#Metrics
precision, recall, f1, accuracy, auc = get_perf_metrics(y_test, svm_pred, svm_proba)

#Confusion Matrix
save_conf_matrix(y_test, svm_pred, '../assets/svm_confusion_matrix.jpg')

#SHAP Values
save_shap(svm.predict, x_test, "svm")

#Hyperparameter optimization graph
pyplot.figure()
pyplot.title("Gamma coefficient vs f1 score")
pyplot.xlabel("Gamma")
pyplot.ylabel("F1 score")
pyplot.scatter(gamma_values, svm_f1_scores)
pyplot.savefig("../assets/svm_hyperparameter_optimization.jpg")

print("Optimized SVM")
print("Precision: {}, Recall: {}, F1: {}, Accuracy: {}, AUC: {}".format(precision, recall, f1, accuracy, auc))

