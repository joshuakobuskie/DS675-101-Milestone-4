# Gets the confusion matric from the true and predicted values
def get_confusion_matrix(y_true, y_pred):
    # Find all unique values
    labels = sorted(set(y_true) | set(y_pred))

    # Initialize matrix
    confusion_matrix = [[0 for i in labels] for j in labels]

    # Fill in matrix
    for true_val, pred_val in zip(y_true, y_pred):
        confusion_matrix[true_val-labels[0]][pred_val-labels[0]] += 1
    
    return confusion_matrix

def get_true_false_positives_negatives(confusion_matrix, class_labels):
    
    truth_table = {class_label: {"true_positives": 0, "false_positives": 0, "true_negatives": 0, "false_negatives": 0} for class_label in class_labels}

    for class_index in class_labels:
        class_index -= class_labels[0]
        true_positives = confusion_matrix[class_index][class_index]
        false_positives = sum(confusion_matrix[row][class_index] for row in range(len(class_labels))) - true_positives
        false_negatives = sum(confusion_matrix[class_index]) - true_positives
        true_negatives = sum(sum(confusion_matrix[row][col] for col in range(len(class_labels))) for row in range(len(class_labels))) - (true_positives + false_positives + false_negatives)

        class_index += class_labels[0]
        truth_table[class_index]["true_positives"] = true_positives
        truth_table[class_index]["false_positives"] = false_positives
        truth_table[class_index]["true_negatives"] = true_negatives
        truth_table[class_index]["false_negatives"] = false_negatives

    return truth_table
    

def calculate_metrics(true_positive, false_positive, true_negative, false_negative):
    
    positives = true_positive + false_negative
    negatives = true_negative + false_positive

    true_positive_rate = true_positive / positives
    true_negative_rate = true_negative / negatives
    false_positive_rate = false_positive / negatives
    false_negative_rate = false_negative / positives

    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    accuracy = (true_positive + true_negative) / (positives + negatives)
    error_rate = (false_positive + false_negative) / (positives + negatives)

    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
    true_skill_score = (true_positive / (true_positive + false_negative)) - (false_positive / (false_positive+true_negative))
    heidke_skill_score = (2 * (true_positive * true_negative - false_positive * false_negative)) / ((true_positive + false_negative) * (false_negative + true_negative) + (true_positive + false_positive) * (false_positive + true_negative))

    return {"positives": positives, "negatives": negatives, "true_positive_rate": true_positive_rate, "true_negative_rate": true_negative_rate, "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate, "recall": recall, "precision": precision, "f1_score": f1_score, "accuracy": accuracy, "error_rate": error_rate, "balanced_accuracy": balanced_accuracy, "true_skill_score": true_skill_score, "heidke_skill_score": heidke_skill_score}

# Converts the multiclass results into aggregate results for model comparison
def get_aggregate_metrics(metrics):
    aggregate_metrics = {key: 0.0 for key in metrics[next(iter(metrics.keys()))]}
    for key in metrics.keys():
        for metric in metrics[key]:
            aggregate_metrics[metric] += metrics[key][metric]

    for metric in aggregate_metrics:
        aggregate_metrics[metric] /= len(metrics)

    return aggregate_metrics

# Still need the Brier score, Brier skill score, AUC, and SHAP values

# Gets the aggregate brier score
def get_brier_score(y_true, y_prob):
    brier_score = 0

    for i in range(len(y_true)):
        for j in range(len(y_prob)):
            if y_true[i] == j:
                actual = 1
            else:
                actual = 0

            brier_score += (y_prob[i][j] - actual) ** 2

    return brier_score / len(y_true)

def get_performance_metrics(y_true, y_pred, y_prob):
    labels = sorted(set(y_true) | set(y_pred))

    confusion_matrix = get_confusion_matrix(y_true, y_pred)

    performance_metrics = get_true_false_positives_negatives(confusion_matrix, labels)

    for class_label in performance_metrics:
        performance_metrics[class_label].update(calculate_metrics(performance_metrics[class_label]["true_positives"], performance_metrics[class_label]["false_positives"], performance_metrics[class_label]["true_negatives"], performance_metrics[class_label]["false_negatives"]))
    
    return(performance_metrics)

import numpy as np
test_data = np.random.randint(0, 10, size=100)
test_proba = np.random.ranf((100, 10))
test_proba /= test_proba.sum(axis=1, keepdims=True)
test_pred = np.argmax(test_proba, axis=1)
#class_metrics = get_performance_metrics(test_data, test_pred, test_pred)
#model_metrics = get_aggregate_metrics(class_metrics)

#print(class_metrics)
#print("#"*128)
#print(model_metrics)
#print("#"*128)

##################################################################
# The above stuff works great, but we could just use this instead?
# This is probably more safe, and we should be able to use it
# We can probably get rid of this and just use below
##################################################################

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
precision = precision_score(test_data, test_pred, average='macro')
recall = recall_score(test_data, test_pred, average='macro')
f1 = f1_score(test_data, test_pred, average='macro')
accuracy = accuracy_score(test_data, test_pred)
auc = roc_auc_score(test_data, test_proba, multi_class='ovr')
conf_matrix = confusion_matrix(test_data, test_pred)

print(precision, recall, f1, accuracy, auc)

import seaborn as sns
import matplotlib.pyplot as plt

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=np.unique(test_data), yticklabels=np.unique(test_data))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('../assets/confusion_matrix.jpg')