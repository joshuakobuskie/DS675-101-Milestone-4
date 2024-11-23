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

def get_true_false_positives_negatives(confusion_matrix):
    classes = len(confusion_matrix)
    
    truth_table = {class_label: {"true_positives": 0, "false_positives": 0, "true_negatives": 0, "false_negatives": 0} for class_label in range(classes)}

    for class_index in range(classes):
        true_positives = confusion_matrix[class_index][class_index]
        false_positives = sum(confusion_matrix[row][class_index] for row in range(classes)) - true_positives
        false_negatives = sum(confusion_matrix[class_index]) - true_positives
        true_negatives = sum(sum(confusion_matrix[row][col] for col in range(classes)) for row in range(classes)) - (true_positives + false_positives + false_negatives)

        truth_table[class_index]["true_positives"] = true_positives
        truth_table[class_index]["false_positives"] = false_positives
        truth_table[class_index]["true_negatives"] = true_negatives
        truth_table[class_index]["false_negatives"] = false_negatives

    return truth_table
    


def get_performance_metrics(y_true, y_pred, y_prob):
    
    confusion_matrix = get_confusion_matrix(y_true, y_pred)

    truth_table = get_true_false_positives_negatives(confusion_matrix)

    print(truth_table)