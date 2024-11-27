from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

def get_perf_metrics(y_test, model_pred, model_proba):
    #Returns precision, recall, f1, accuract, AUC
    return precision_score(y_test, model_pred, average='macro'), recall_score(y_test, model_pred, average='macro'), f1_score(y_test, model_pred, average='macro'), accuracy_score(y_test, model_pred), roc_auc_score(y_test, model_proba, multi_class='ovr')

