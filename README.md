# svm_spam_classification_matlab
This project implements and evaluates Support Vector Machines (SVM) to classify spam email messages using MATLAB. It was developed as part of the EE5904/ME5404 module at the National University of Singapore (NUS).

The goal was to design an optimal SVM classifier by comparing different kernels and parameters. The project involves high-dimensional data processing, mathematical verification of kernel admissibility, and optimization using quadratic programming.

🛠️ Technical Stack

**Language**: MATLAB 
**Toolbox**: Optimization Toolbox (specifically the quadprog function) 
**Algorithms**: Hard Margin SVM, Soft Margin 
**SVM Kernels**: Linear, Polynomial (Degrees 2 through 5)

🧪 Methodology

**Data Pre-processing**: Implemented Standardization (Z-score normalization) to ensure features have a mean of 0 and a standard deviation of 1.
**Kernel Admissibility**: Verified every kernel using Mercer’s Condition by checking if the eigenvalues of the kernel matrix were $\ge 0$.
**Hyperparameter Tuning**: Iterated through different values of the cost parameter ($C$) and polynomial degrees ($p$) to find the balance between margin width and misclassification.
