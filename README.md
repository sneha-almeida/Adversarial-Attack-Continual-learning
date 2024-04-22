# ContinuousPoisoning: Assessing the Claims and Security Aspects of Continual Learning
Run the files in the following order:
1. DefenderEnvironment/continual_learning_defender_env_60000_samples.py
   This python file simulates a defender environment for continual learning with 60,0000 samples of each dataset. So we have 60,000 samples in the MNIST, KMNIST, and FashionMNIST dataset each. This file trains a model for SI, EWC, and Online EWC strategy each for both, RotatedMNIST scenario as well as for MNIST Fellowship scenario. 
3. DefenderEnvironment/continual_learning_defender_env_2400_samples.py
4. DefenderEnvironment/continual_learning_defender_env_1200_samples.py
5. LabelFlippingEnvironment/continual_learning_label_flipping.py
6. AttackerEnvironment/1200Samples/rotated_mnist_adv_attack_10_percent.py
7. AttackerEnvironment/1200Samples/rotated_mnist_adv_attack_15_percent.py
8. AttackerEnvironment/1200Samples/rotated_mnist_adv_attack_20_percent.py
9. AttackerEnvironment/1200Samples/rotated_mnist_adv_attack_5_percent.py
10. AttackerEnvironment/2400Samples/rotated_mnist_adv_attack_10_percent_2400_samples.py
11. AttackerEnvironment/2400Samples/rotated_mnist_adv_attack_15_percent_2400_samples.py
12. AttackerEnvironment/2400Samples/rotated_mnist_adv_attack_20_percent_2400_samples.py
13. AttackerEnvironment/2400Samples/rotated_mnist_adv_attack_5_percent_2400_samples.py

To execute all these Python files, use the command "python filename.py". For example, python continual_learning_defender_env_60000_samples.py

Files numbered from 1 to 4 will generate 6 trained models and 2 CSV files each. The CSV files will summarize the performance of these models. 

Files numbered from 5 to 12 will generate 3 trained models and 1 CSV file each. The CSV files will summarize the performance of these models.

