# Imports
import os
from appscanner.preprocessor import Preprocessor
import scapy

def files_labels(directory):
    files = []


    labels = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
            labels.append(f.split('_')[0])
    return files, labels


########################################################################
#                             Handle input                             #
########################################################################
# Get file paths and labels
# Create Flows and labels
pth = input("Enter the path of folder contains pcap files:")
save_name = input("Enter a name to save flows in a pickle file:") + ".p"

my_files, my_labels = files_labels(pth)

########################################################################
#                              Read data                               #
########################################################################
# Create Preprocessor object
preprocessor = Preprocessor(verbose=True)
# Create Flows and labels
X, y = preprocessor.process(files=my_files, labels=my_labels)

########################################################################
#                             Save Extracted features                  #
########################################################################
# Save flows and labels to file
preprocessor.save(save_name, X, y)
# Appscanner_CrossMarket_Intersection_Apps