# ContinuousPoisoning: Assessing the Claims and Security Aspects of Continual Learning

In this work, we have reproduced the results in the paper titled "Targeted Data Poisoning Attacks Against Continual Learning Neural Networks" by implementing it. 

Run the files in the following order:
1. DefenderEnvironment/continual_learning_defender_env_60000_samples.py: 
   This Python file simulates a defender environment for continual learning with 60,0000 samples of each dataset. So we have 60,000 samples in the MNIST, KMNIST, and FashionMNIST datasets each. This file trains a model for SI, EWC, and Online EWC strategy each for both, the RotatedMNIST scenario and the MNIST Fellowship scenario.
   
3. DefenderEnvironment/continual_learning_defender_env_2400_samples.py: This Python file simulates a defender environment for continual learning with 2400 samples of each dataset. So we have 2400 samples in the MNIST, KMNIST, and FashionMNIST datasets each. This file trains a model for SI, EWC, and Online EWC strategy each for both, the RotatedMNIST scenario and the MNIST Fellowship scenario.
   
5. DefenderEnvironment/continual_learning_defender_env_1200_samples.py: This Python file simulates a defender environment for continual learning with 2400 samples of each dataset. So we have 1200 samples in the MNIST, KMNIST, and FashionMNIST datasets each. This file trains a model for SI, EWC, and Online EWC strategy each for both, the RotatedMNIST scenario and the MNIST Fellowship scenario.
   
7. LabelFlippingEnvironment/continual_learning_label_flipping.py: This Python file simulates a label-flipping attack environment for continual learning with 60,000 samples of each dataset. So we have 60,0000 samples in the MNIST, KMNIST, and FashionMNIST datasets each. This file trains a model poisoned by label-flipping attack for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario and the MNIST Fellowship scenario.
   
9. AttackerEnvironment/1200Samples/rotated_mnist_adv_attack_10_percent.py: This Python file simulates an adversarial attack for continual learning with 1200 samples of MNIST dataset. This file trains a model poisoned by the adversarial attack algorithm with 10% poisoning as proposed in 'Targeted Data Poisoning Attacks Against Continual Learning Neural Networks' for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario. 
    
11. AttackerEnvironment/1200Samples/rotated_mnist_adv_attack_15_percent.py: This Python file simulates an adversarial attack for continual learning with 1200 samples of MNIST dataset. This file trains a model poisoned by the adversarial attack algorithm with 15% poisoning as proposed in 'Targeted Data Poisoning Attacks Against Continual Learning Neural Networks' for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario.
    
13. AttackerEnvironment/1200Samples/rotated_mnist_adv_attack_20_percent.py: This Python file simulates an adversarial attack for continual learning with 1200 samples of MNIST dataset. This file trains a model poisoned by the adversarial attack algorithm with 20% poisoning as proposed in 'Targeted Data Poisoning Attacks Against Continual Learning Neural Networks' for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario.
    
15. AttackerEnvironment/1200Samples/rotated_mnist_adv_attack_5_percent.py: This Python file simulates an adversarial attack for continual learning with 1200 samples of MNIST dataset. This file trains a model poisoned by the adversarial attack algorithm with 5% poisoning as proposed in 'Targeted Data Poisoning Attacks Against Continual Learning Neural Networks' for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario.
    
17. AttackerEnvironment/2400Samples/rotated_mnist_adv_attack_10_percent_2400_samples.py: This Python file simulates an adversarial attack for continual learning with 2400 samples of MNIST dataset. This file trains a model poisoned by the adversarial attack algorithm with 10% poisoning as proposed in 'Targeted Data Poisoning Attacks Against Continual Learning Neural Networks' for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario.
    
19. AttackerEnvironment/2400Samples/rotated_mnist_adv_attack_15_percent_2400_samples.py: This Python file simulates an adversarial attack for continual learning with 2400 samples of MNIST dataset. This file trains a model poisoned by the adversarial attack algorithm with 15% poisoning as proposed in 'Targeted Data Poisoning Attacks Against Continual Learning Neural Networks' for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario.
    
21. AttackerEnvironment/2400Samples/rotated_mnist_adv_attack_20_percent_2400_samples.py: This Python file simulates an adversarial attack for continual learning with 2400 samples of MNIST dataset. This file trains a model poisoned by the adversarial attack algorithm with 20% poisoning as proposed in 'Targeted Data Poisoning Attacks Against Continual Learning Neural Networks' for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario.
    
23. AttackerEnvironment/2400Samples/rotated_mnist_adv_attack_5_percent_2400_samples.py: This Python file simulates an adversarial attack for continual learning with 2400 samples of MNIST dataset. This file trains a model poisoned by the adversarial attack algorithm with 5% poisoning as proposed in 'Targeted Data Poisoning Attacks Against Continual Learning Neural Networks' for SI, EWC, and Online EWC strategy each for the RotatedMNIST scenario.

To execute all these Python files, use the command "python filename.py". For example, python continual_learning_defender_env_60000_samples.py

Files from 1 to 4 will generate 6 trained models and 2 CSV files each. The CSV files will summarize the performance of these models. 

Files from 5 to 12 will generate 3 trained models and 1 CSV file each. The CSV files will summarize the performance of these models.

