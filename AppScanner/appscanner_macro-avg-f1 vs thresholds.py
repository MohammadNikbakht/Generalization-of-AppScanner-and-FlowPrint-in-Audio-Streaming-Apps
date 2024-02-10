import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from appscanner.appscanner import AppScanner
from appscanner.preprocessor import Preprocessor
from sklearn.metrics import f1_score


########################################################################
#                              Read data                               #
########################################################################
preprocessor = Preprocessor(verbose=True)
X_train, y_train = preprocessor.load('Select the training dataset')
X_test, y_test = preprocessor.load('Select the testing dataset')

########################################################################
#                 Modify Labels in Original Test Data                  #
########################################################################
y_test[:] = 'unknown'

########################################################################
#                     Split Training Data (70/30)                      #
########################################################################
X_train_new, X_additional_test, y_train_new, y_additional_test = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42)

########################################################################
#               Combine Additional Test Data with Test Set              #
########################################################################
X_test_combined = np.concatenate((X_test, X_additional_test))
y_test_combined = np.concatenate((y_test, y_additional_test))

########################################################################
#                          Scale Features                              #
########################################################################
scaler = MinMaxScaler()
X_train_new = scaler.fit_transform(X_train_new)
X_test_combined = scaler.transform(X_test_combined)

########################################################################
#                              AppScanner                              #
########################################################################


# Unique labels including 'unknown'
l1 = np.unique(np.concatenate((y_train, ['unknown'])))

thresholds = ["Enter thresholds"]
macro_avg_f1 = []

for threshold in thresholds:
    scanner = AppScanner(threshold=threshold)
    scanner.fit(X_train_new, y_train_new)

    y_pred = scanner.predict(X_test_combined)

    ########################################################################
    #                           Print Evaluation                           #
    ########################################################################

    f1_macro = (f1_score(y_true=y_test_combined, y_pred=y_pred, labels=l1, average='macro'))
    macro_avg_f1.append(round(float(f1_macro) * 100, 1))  # Converting to percentage with one decimal place


plt.figure(figsize=(10, 6))
plt.plot(thresholds, macro_avg_f1, marker='o', markersize=8, linewidth=2, linestyle='-', color='b')
plt.xlabel('Thresholds', fontsize=12)
plt.ylabel('F1 Macro Avg (%)', fontsize=12)
plt.xticks(thresholds,fontsize=10)
plt.yticks(fontsize=10)

for i in range(len(thresholds)):
    plt.annotate(str(macro_avg_f1[i]), xy=(thresholds[i], macro_avg_f1[i]), xytext=(0, 8),
                 textcoords='offset points', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()