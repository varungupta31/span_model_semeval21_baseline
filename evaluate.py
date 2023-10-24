import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json

LABELS_2021 = [
    "Appeal to authority",
    "Appeal to fear/prejudice",
    "Black-and-white Fallacy/Dictatorship",
    "Causal Oversimplification",
    "Doubt",
    "Exaggeration/Minimisation",
    "Flag-waving",
    "Glittering generalities (Virtue)",
    "Loaded Language",
    "Misrepresentation of Someone's Position (Straw Man)",
    "Name calling/Labeling",
    "Obfuscation, Intentional vagueness, Confusion",
    "Presenting Irrelevant Data (Red Herring)",
    "Reductio ad hitlerum",
    "Repetition",
    "Slogans",
    "Smears",
    "Thought-terminating cliché",
    "Whataboutism",
    "Bandwagon"
]

pred_file_path = r'/home2/varungupta/span_model_semeval21_baseline/task2_prediction.json'
gt_file_path = r'/ssd_scratch/cvit/varun/SEMEVAL-2021-task6-corpus/data/test_set_task2.txt'

pred_file = open(pred_file_path, 'r')
gt_file = open(gt_file_path, 'r')
pred = json.load(pred_file)
gt = json.load(gt_file)

data_df = pd.DataFrame(gt)
data_df['data_label_indices'] = data_df['labels'].apply(lambda x: ', '.join([label['technique'] for label in x]))
data_df['data_labels'] = data_df['labels'].apply(lambda x: [LABELS_2021.index(label['technique']) for label in x])
data_df['data_labels_onehot'] = data_df['data_labels'].apply(lambda x: np.array([int(i in x) for i in range(len(LABELS_2021))]))

# Create a DataFrame for 'pred'
pred_df = pd.DataFrame(pred)
pred_df['pred_label_indices'] = pred_df['labels'].apply(lambda x: ', '.join([label['technique'] for label in x]))
pred_df['pred_labels'] = pred_df['labels'].apply(lambda x: [LABELS_2021.index(label['technique']) for label in x])
pred_df['pred_labels_onehot'] = pred_df['pred_labels'].apply(lambda x: np.array([int(i in x) for i in range(len(LABELS_2021))]))

# Merge the two DataFrames on 'id'
result_df = data_df.merge(pred_df, on='id')

# Rename columns
result_df.rename(columns={"data_labels_onehot": "y_true", "pred_labels_onehot": "y_pred_t"}, inplace=True)
#result_df.rename(columns={'data_label_indices': 'data_indices', 'data_labels_onehot': 'gt_labels', 'pred_label_indices': 'pred_indices', 'pred_labels_onehot': 'pred_labels'}, inplace=True)
result_df.head()

y_true = np.array(result_df['y_true'].tolist())
y_pred_t = np.array(result_df['y_pred_t'].tolist())

F1_Micro = f1_score(y_true, y_pred_t, average='micro')
F1_Macro = f1_score(y_true, y_pred_t, average='macro')
acc = accuracy_score(y_true, y_pred_t)
premi = precision_score(y_true, y_pred_t, average='micro')
prema = precision_score(y_true, y_pred_t, average='macro')
recmi = recall_score(y_true, y_pred_t, average='micro')
recma = recall_score(y_true, y_pred_t, average='macro')

print("F1 Micro: ", F1_Micro)
print("F1 Macro: ", F1_Macro)
print("Accuracy: ", acc)
print("Precision Micro: ", premi)
print("Precision Macro: ", prema)
print("Recall Micro: ", recmi)
print("Recall Macro: ", recma)