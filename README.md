**This repository contains the code for evaluating the generalization of AppScanner[1] and FlowPrint[2] in identifying audio streaming applications.**


## Introduction
AppScanner and Flowprint are two effective methods for fingerprinting mobile apps from encrypted network traffic. We evaluated the generalization of these two methods in identifying audio streaming apps. To accomplish this, we implemented AppScanner and Flowprint in various network environments. The evaluation process involved validating the methods and conducting three separate experiments. In the validation phase, we used a comprehensive and exclusive dataset specifically curated for identifying audio streaming apps within network traffic. Next, in the experiments phase, we evaluated the performance of these methods with datasets that had completely different characteristics from the training data.
More information: https://doi.org/10.1109/ICCKE65377.2024.10874577



## Refrences
[1] V. F. Taylor, R. Spolaor, M. Conti, and I. Martinovic, "Appscanner: Automatic fingerprinting of smartphone apps from encrypted network traffic," in 2016 IEEE European Symposium on Security and Privacy (EuroS&P), 2016: IEEE, pp. 439-454.

[2] T. Van Ede et al, "Flowprint: Semi-supervised mobile-app fingerprinting on encrypted network traffic," in Network and distributed system security symposium (NDSS), 2020, vol. 27.
