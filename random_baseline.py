import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
np.random.seed(31)
# Load the data from the CSV file
data = pd.read_csv('pred_result.csv')

# Define the number of random runs
num_runs = 1
F1_Micro_list = []
F1_Macro_list = []
acc_list = []
premi_list = []
prema_list = []
recmi_list = []
recma_list = []
hamming_loss_list = []

# Loop over the random runs
for _ in range(num_runs):
    random_predictions = np.random.randint(2, size=(len(data), 20))  # Generate random binary predictions (0 or 1)
    
    # Convert the random predictions to the same format as y_true
    random_predictions = [list(row) for row in random_predictions]

    # Extract the true labels and random predictions for each example
    y_true = [list(map(int, row.strip('[]').split())) for row in data['y_true']]
    y_true = np.array(y_true)
    random_predictions = np.array(random_predictions)
    print(y_true)
    print(random_predictions)
    # Calculate precision and recall for each class
    F1_Micro = f1_score(y_true, random_predictions, average='micro')
    acc = accuracy_score(y_true, random_predictions)
    premi = precision_score(y_true, random_predictions, average='micro')
    prema = precision_score(y_true, random_predictions, average='macro')
    recmi = recall_score(y_true, random_predictions, average='micro')
    recma = recall_score(y_true, random_predictions, average='macro')
    hamming = hamming_loss(y_true, random_predictions)
    
    # Append the scores to the lists
    F1_Micro_list.append(F1_Micro)
    F1_Macro_list.append(F1_Macro)
    acc_list.append(acc)
    premi_list.append(premi)
    prema_list.append(prema)
    recmi_list.append(recmi)
    recma_list.append(recma)
    hamming_loss_list.append(hamming)


# Calculate the average precision and recall
# average_F1_Micro = np.mean(F1_Micro_list)
# average_F1_Macro = np.mean(F1_Macro_list)
# average_acc = np.mean(acc_list)
# average_premi = np.mean(premi_list)
# average_prema = np.mean(prema_list)
# average_recmi = np.mean(recmi_list)
# average_recma = np.mean(recma_list)

# average_F1_Micro = np.max(F1_Micro_list)
# average_F1_Macro = np.max(F1_Macro_list)
# average_acc = np.max(acc_list)
# average_premi = np.max(premi_list)
# average_prema = np.max(prema_list)
# average_recmi = np.max(recmi_list)
# average_recma = np.max(recma_list)
# hamming_loss = np.min(hamming_loss_list)


print('Average F1 Micro: {:.5f}'.format(average_F1_Micro))
print('Average F1 Macro: {:.5f}'.format(average_F1_Macro))
print('Average Accuracy: {:.5f}'.format(average_acc))
print('Average Precision Micro: {:.5f}'.format(average_premi))
print('Average Precision Macro: {:.5f}'.format(average_prema))
print('Average Recall Micro: {:.5f}'.format(average_recmi))
print('Average Recall Macro: {:.5f}'.format(average_recma))
print('Average Hamming Loss: {:.5f}'.format(hamming_loss))


