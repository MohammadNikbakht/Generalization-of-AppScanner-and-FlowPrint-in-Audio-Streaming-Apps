import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from flowprint.flowprint import FlowPrint
from flowprint.preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



#------------------------------------------------- Model Validation -----------------------------------------------------


########################################################################
#                              Read data                               #
########################################################################
# Create Preprocessor object
preprocessor = Preprocessor(verbose=True)
# Load train and test data separately
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
#                              flowprint                               #
########################################################################
# Create FlowPrint object
flowprint = FlowPrint(
    batch       = 300,
    window      = 30,
    correlation = 0.1,
    similarity  = 0.9
)

# Fit FlowPrint with flows and labels
flowprint.fit(X_train_new, y_train_new)

# Create fingerprints for test data
fp_test = flowprint.fingerprint(X_test_combined)
# Predict best matching fingerprints for each test fingerprint

# Recognise which app produced each flow
y_recognize = flowprint.recognize(fp_test)

# Detect previously unseen apps
# +1 if a flow belongs to a known app, -1 if a flow belongs to an unknown app
y_detect = flowprint.detect(fp_test, threshold= 0.40)

#save results in list
results = []
for i in range(len(X_test_combined)):
    sample_results = {
        "Sample": i + 1,
        "True Class": y_test_combined[i],
        "y_recognize": y_recognize[i],
        "y_detect": y_detect[i]
    }
    results.append(sample_results)

#Change y_recognize based on y_detect in results_list
for i in range(len(results)):
    if results[i]['y_detect'] == -1:
        results[i]['y_recognize'] = 'unknown'

# convert results to tuple
results_tuple = tuple(results)

# extract y_recognize & y_test from tuple
y_test_final = [result['True Class'] for result in results_tuple]
y_recognize_final = [result['y_recognize'] for result in results_tuple]


########################################################################
#                              Define Classes                          #
########################################################################
classes = np.unique(np.concatenate((y_train, ['unknown'])))


########################################################################
#                           Print Evaluation                           #
########################################################################
print("Classification Report of Validation:")
print(classification_report(y_true=y_test_final, y_pred=y_recognize_final, labels=classes, digits=4))

# Plot the confusion matrix
confusion = confusion_matrix(y_true=y_test_final, y_pred=y_recognize_final, labels=classes)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("FlowPrint Confusion Matrix",fontsize=14)
plt.xlabel('Predicted Labels',fontsize=14)
plt.ylabel('True Labels',fontsize=14)
plt.show()



#----------------------------------------------- Experiment 1 -------------------------------------------------------


########################################################################
#                              Read data                               #
########################################################################
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
#                              flowprint                               #
########################################################################
# Create fingerprints for test data
fp_test = flowprint.fingerprint(X_test_combined)
# Predict best matching fingerprints for each test fingerprint


# Recognise which app produced each flow
y_recognize = flowprint.recognize(fp_test)
#print(y_recognize)

# Detect previously unseen apps
# +1 if a flow belongs to a known app, -1 if a flow belongs to an unknown app
y_detect = flowprint.detect(fp_test, threshold= 0.40)

#save results in list
results = []
for i in range(len(X_test_combined)):
    sample_results = {
        "Sample": i + 1,
        "True Class": y_test_combined[i],
        "y_recognize": y_recognize[i],
        "y_detect": y_detect[i]
    }
    results.append(sample_results)

#Change y_recognize based on y_detect in results_list
for i in range(len(results)):
    if results[i]['y_detect'] == -1:
        results[i]['y_recognize'] = 'unknown'

# convert results to tuple
results_tuple = tuple(results)

# extract y_recognize & y_test from tuple
y_test_final = [result['True Class'] for result in results_tuple]
y_recognize_final = [result['y_recognize'] for result in results_tuple]


########################################################################
#                           Print Evaluation                           #
########################################################################
print("Classification Report of Test 1:")
print(classification_report(y_true=y_test_final, y_pred=y_recognize_final, labels=classes, digits=4))

# Plot the confusion matrix
confusion = confusion_matrix(y_true=y_test_final, y_pred=y_recognize_final, labels=classes)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("FlowPrint Confusion Matrix-Test 1",fontsize=14)
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
#                              flowprint                               #
########################################################################
# Create fingerprints for test data
fp_test = flowprint.fingerprint(X_test)
# Predict best matching fingerprints for each test fingerprint


# Recognise which app produced each flow
y_recognize = flowprint.recognize(fp_test)
#print(y_recognize)

# Detect previously unseen apps
# +1 if a flow belongs to a known app, -1 if a flow belongs to an unknown app
y_detect = flowprint.detect(fp_test, threshold= 0.40)

#save results in list
results = []
for i in range(len(X_test)):
    sample_results = {
        "Sample": i + 1,
        "True Class": y_test[i],
        "y_recognize": y_recognize[i],
        "y_detect": y_detect[i]
    }
    results.append(sample_results)

#Change y_recognize based on y_detect in results_list
for i in range(len(results)):
    if results[i]['y_detect'] == -1:
        results[i]['y_recognize'] = 'unknown'

# convert results to tuple
results_tuple = tuple(results)

# extract y_recognize & y_test from tuple
y_test_final = [result['True Class'] for result in results_tuple]
y_recognize_final = [result['y_recognize'] for result in results_tuple]


########################################################################
#                           Print Evaluation                           #
########################################################################
print("Classification Report of Test 2:")
print(classification_report(y_true=y_test_final, y_pred=y_recognize_final, labels=classes, digits=4))

# Plot the confusion matrix
confusion = confusion_matrix(y_true=y_test_final, y_pred=y_recognize_final, labels=classes)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("FlowPrint Confusion Matrix-Test 2",fontsize=14)
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
#                              flowprint                               #
########################################################################
# Create fingerprints for test data
fp_test = flowprint.fingerprint(X_test)
# Predict best matching fingerprints for each test fingerprint


# Recognise which app produced each flow
y_recognize = flowprint.recognize(fp_test)
#print(y_recognize)

# Detect previously unseen apps
# +1 if a flow belongs to a known app, -1 if a flow belongs to an unknown app
y_detect = flowprint.detect(fp_test, threshold= 0.40)

#save results in list
results = []
for i in range(len(X_test)):
    sample_results = {
        "Sample": i + 1,
        "True Class": y_test[i],
        "y_recognize": y_recognize[i],
        "y_detect": y_detect[i]
    }
    results.append(sample_results)

#Change y_recognize based on y_detect in results_list
for i in range(len(results)):
    if results[i]['y_detect'] == -1:
        results[i]['y_recognize'] = 'unknown'

# convert results to tuple
results_tuple = tuple(results)

# extract y_recognize & y_test from tuple
y_test_final = [result['True Class'] for result in results_tuple]
y_recognize_final = [result['y_recognize'] for result in results_tuple]


########################################################################
#                           Print Evaluation                           #
########################################################################
print("Classification Report of Test 3:")
print(classification_report(y_true=y_test_final, y_pred=y_recognize_final, labels=classes, digits=4))

# Plot the confusion matrix
confusion = confusion_matrix(y_true=y_test_final, y_pred=y_recognize_final, labels=classes)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("FlowPrint Confusion Matrix-Test 3",fontsize=14)
plt.xlabel('Predicted Labels',fontsize=14)
plt.ylabel('True Labels',fontsize=14)
plt.show()
