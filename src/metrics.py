from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from matplotlib import pyplot
import seaborn as sns

def get_perf_metrics(y_test, model_pred, model_proba):
    #Returns precision, recall, f1, accuract, AUC
    return precision_score(y_test, model_pred, average='macro'), recall_score(y_test, model_pred, average='macro'), f1_score(y_test, model_pred, average='macro'), accuracy_score(y_test, model_pred), roc_auc_score(y_test, model_proba, multi_class='ovr')

def save_conf_matrix(y_test, model_pred, destination):
    #Saves confusion matrix in the destination
    conf_matrix = confusion_matrix(y_test, model_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    pyplot.title("Confusion Matrix")
    pyplot.xlabel("Predicted Label")
    pyplot.ylabel("True Label")
    pyplot.tight_layout()
    pyplot.savefig(destination)
    pyplot.clf()
    pyplot.close()