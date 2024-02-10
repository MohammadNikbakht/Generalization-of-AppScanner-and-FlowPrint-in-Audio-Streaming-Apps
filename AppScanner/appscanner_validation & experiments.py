import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from appscanner.appscanner import AppScanner
from appscanner.preprocessor import Preprocessor


#------------------------------------------------- Model Validation -----------------------------------------------------


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
scanner = AppScanner(threshold=0.95)
scanner.fit(X_train_new, y_train_new)
y_pred = scanner.predict(X_test_combined)


########################################################################
#                              Define Classes                          #
########################################################################
l1 = np.unique(np.concatenate((y_train, ['unknown'])))


########################################################################
#                           Print Evaluation                           #
########################################################################
print("Classification Report of Validation:")
print(classification_report(y_true=y_test_combined, y_pred=y_pred, labels=l1, digits=4))

# Plot the confusion matrix
confusion = confusion_matrix(y_true=y_test_combined, y_pred=y_pred, labels=l1)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=l1, yticklabels=l1, annot_kws={"size": 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("AppScanner Confusion Matrix-Validation",fontsize=14)
plt.xlabel('Predicted Labels',fontsize=14)
plt.ylabel('True Labels',fontsize=14)
plt.show()


#----------------------------------------------- Experiment 1 -------------------------------------------------------


X_val, y_val = preprocessor.load('Select the testing dataset')
X_test, y_test = preprocessor.load('Select the testing dataset')


########################################################################
#                 Modify Labels in Original Test Data                  #
########################################################################
y_test[:] = 'unknown'


########################################################################
#               Combine Additional Test Data with Test Set              #
########################################################################
X_test_combined = np.concatenate((X_test, X_val))
y_test_combined = np.concatenate((y_test, y_val))

########################################################################
#                          Scale Features                              #
########################################################################
X_test_combined = scaler.transform(X_test_combined)


########################################################################
#                              AppScanner                              #
########################################################################
y_pred = scanner.predict(X_test_combined)


########################################################################
#                           Print Evaluation                           #
########################################################################
print("Classification Report of Test 1:")
print(classification_report(y_true=y_test_combined, y_pred=y_pred, labels=l1, digits=4))

# Plot the confusion matrix
confusion = confusion_matrix(y_true=y_test_combined, y_pred=y_pred, labels=l1)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=l1, yticklabels=l1, annot_kws={"size": 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("AppScanner Confusion Matrix-Test 1",fontsize=14)
plt.xlabel('Predicted Labels',fontsize=14)
plt.ylabel('True Labels',fontsize=14)
plt.show()


#----------------------------------------------- Experiment 2 -------------------------------------------------------


########################################################################
#                              Read data                               #
########################################################################
X_test, y_test = preprocessor.load('Select the testing dataset')


########################################################################
#                 Modify Labels in Original Test Data                  #
########################################################################
y_test[:] = 'unknown'


########################################################################
#                          Scale Features                              #
########################################################################
X_test = scaler.transform(X_test)


########################################################################
#                              AppScanner                              #
########################################################################
y_pred = scanner.predict(X_test)


########################################################################
#                           Print Evaluation                           #
########################################################################
print("Classification Report of Test 2:")
print(classification_report(y_true=y_test, y_pred=y_pred, labels=l1, digits=4))

# Plot the confusion matrix
confusion = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=l1)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=l1, yticklabels=l1, annot_kws={"size": 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("AppScanner Confusion Matrix-Test 2",fontsize=14)
plt.xlabel('Predicted Labels',fontsize=14)
plt.ylabel('True Labels',fontsize=14)
plt.show()



#----------------------------------------------- Experiment 3 -------------------------------------------------------


########################################################################
#                              Read data                               #
########################################################################
X_test, y_test = preprocessor.load('Select the testing dataset')


########################################################################
#                 Modify Labels in Original Test Data                  #
########################################################################
y_test[:] = 'unknown'


########################################################################
#                          Scale Features                              #
########################################################################
X_test = scaler.transform(X_test)

########################################################################
#                              AppScanner                              #
########################################################################
y_pred = scanner.predict(X_test)


########################################################################
#                           Print Evaluation                           #
########################################################################
print("Classification Report of Test 3:")
print(classification_report(y_true=y_test, y_pred=y_pred, labels=l1, digits=4))

# Plot the confusion matrix
confusion = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=l1)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=l1, yticklabels=l1, annot_kws={"size": 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("AppScanner Confusion Matrix-Test 3",fontsize=14)
plt.xlabel('Predicted Labels',fontsize=14)
plt.ylabel('True Labels',fontsize=14)
plt.show()