import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from flowprint.flowprint import FlowPrint
from flowprint.preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



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



thresholds = ["Enter thresholds"]
macro_avg_f1 = []


for threshold in thresholds:

    # Recognise which app produced each flow
    y_recognize = flowprint.recognize(fp_test)
    #print(y_recognize)


    # Detect previously unseen apps
    # +1 if a flow belongs to a known app, -1 if a flow belongs to an unknown app
    y_detect = flowprint.detect(fp_test, threshold= threshold)

    results = []
    for i in range(len(X_test_combined)):
        sample_results = {
            "Sample": i + 1,
            "True Class": y_test_combined[i],
            "y_recognize": y_recognize[i],
            "y_detect": y_detect[i]
        }
        results.append(sample_results)

        #print(results)

    # Change y_recognize based on y_detect in results_list
    for i in range(len(results)):
        if results[i]['y_detect'] == -1:
            results[i]['y_recognize'] = 'unknown'

    # convert results to tuple
    results_tuple = tuple(results)
    #print(results_tuple)

    # extract y_recognize & y_test from tuple
    y_test_final = [result['True Class'] for result in results_tuple]
    y_recognize_final = [result['y_recognize'] for result in results_tuple]

    # Get unique classes
    classes = np.unique(np.concatenate((y_test_final, y_recognize_final)))

    f1_macro = (f1_score(y_true=y_test_final, y_pred=y_recognize_final, labels=classes, average='macro'))
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