from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from matplotlib import pyplot
import seaborn as sns
import shap

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

def save_shap(model_predict, x_test, model_name):
    #This should be provided with model.predict, x_test, and a string that contains the model name
    explainer = shap.Explainer(model_predict, x_test)
    shap_values = explainer(x_test)

    shap.plots.bar(shap_values, show=False)
    pyplot.savefig("../assets/"+model_name+"_shap_values_bar.jpg")
    pyplot.clf()
    pyplot.close()

    shap.summary_plot(shap_values, show=False)
    pyplot.savefig("../assets/"+model_name+"_shap_values_summary.jpg")
    pyplot.clf()
    pyplot.close()

    shap.summary_plot(shap_values, plot_type='violin', show=False)
    pyplot.savefig("../assets/"+model_name+"_shap_values_violin.jpg")
    pyplot.clf()
    pyplot.close()