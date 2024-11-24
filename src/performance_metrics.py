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
    

def calculate_metrics(true_positive, false_positive, true_negative, false_negative):
    
    #Calculate performance metrics based on class slides
    positives = true_positive + false_negative
    negatives = true_negative + false_positive

    true_positive_rate = true_positive / positives
    true_negative_rate = true_negative / negatives
    false_positive_rate = false_positive / negatives
    false_negative_rate = false_negative / positives

    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)

    #F1 can encounter 0 division error
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    accuracy = (true_positive + true_negative) / (positives + negatives)
    error_rate = (false_positive + false_negative) / (positives + negatives)

    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
    true_skill_score = (true_positive / (true_positive + false_negative)) - (false_positive / (false_positive+true_negative))
    heidke_skill_score = (2 * (true_positive * true_negative - false_positive * false_negative)) / ((true_positive + false_negative) * (false_negative + true_negative) + (true_positive + false_positive) * (false_positive + true_negative))

    brier_score = 0 # temporary while solving

    base_prob = 0 # temporary while solving
    base_brier_score = 0 #temporary while solving
    brier_skill_score = 1 - (brier_score / base_brier_score)

    auc = 0 #temporary while solving

    return [positives, negatives, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, recall, precision, f1_score, accuracy, error_rate, balanced_accuracy, true_skill_score, heidke_skill_score, brier_score, brier_skill_score, auc]


def get_performance_metrics(y_true, y_pred, y_prob):
    
    confusion_matrix = get_confusion_matrix(y_true, y_pred)

    performance_metrics = get_true_false_positives_negatives(confusion_matrix)

    for class_label in performance_metrics:
        performance_metrics[class_label].update(calculate_metrics(performance_metrics[class_label]["true_positives"], performance_metrics[class_label]["false_positives"], performance_metrics[class_label]["true_negatives"], performance_metrics[class_label]["false_negatives"]))
    
    print(performance_metrics)
