from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

# Gets the confusion matric from the true and predicted values
def get_confusion_matrix(y_true, y_pred):

    # Find all unique values
    labels = sorted(set(y_true) | set(y_pred))

    # Initialize matrix
    confusion_matrix = [[0 for i in labels] for j in labels]

    # Fill in matrix
    for true_val, pred_val in zip(y_true, y_pred):
        confusion_matrix[true_val][pred_val] += 1
    
    return confusion_matrix

def get_performance_metrics(y_true, y_pred, y_prob):
    
    confusion_matrix = get_confusion_matrix(y_true, y_pred)
    