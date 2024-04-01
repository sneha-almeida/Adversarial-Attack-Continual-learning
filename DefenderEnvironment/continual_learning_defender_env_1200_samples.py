# -*- coding: utf-8 -*-
"""continual_learning_defender_env.ipynb
"""



"""# Model Definition"""

import torch
import torchvision

"""# Dataset

## Scenario 1: Domain Incremental - RotatedMNIST
"""

from avalanche.benchmarks.classic import RotatedMNIST
import pandas as pd

performance_rotated_mnist = pd.DataFrame(index=["SI","EWC","Online EWC"], columns=["Loss - Task 0","Loss - Task 1","Loss - Task 2","Loss - Task 3","Loss - Task 4","Accuracy - Task 0","Accuracy - Task 1","Accuracy - Task 2","Accuracy - Task 3","Accuracy - Task 4","Accuracy", "Loss"])



import random
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from avalanche.benchmarks import dataset_benchmark
import torch.utils.data as data_utils

train_indices = []
test_indices = []

train_indices = random.sample(range(0,60000),1200)
test_indices = random.sample(range(0,10000),200)

rotation_angle_0 = (0.0,0.1)  # Specify the desired rotation angle in degrees
rotation_angle_40 = (40.0,40.1) #(40-42) otherwise it will rotate between +40 and -40 degrees
rotation_angle_80 = (80.0,80.1)
rotation_angle_120 = (120.0,120.1)
rotation_angle_160 = (160.0,160.1)

rotation_transform_0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomRotation(rotation_angle_0),
    transforms.ToTensor()
])

rotation_transform_40 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomRotation(rotation_angle_40),
    transforms.ToTensor()
])

rotation_transform_80 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomRotation(rotation_angle_80),
    transforms.ToTensor()
])

rotation_transform_120 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomRotation(rotation_angle_120),
    transforms.ToTensor()
])

rotation_transform_160 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomRotation(rotation_angle_160),
    transforms.ToTensor()
])

train_mnist_rotated_0 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_0
)
train_mnist_rotated_0 = data_utils.Subset(train_mnist_rotated_0, train_indices)



test_mnist_rotated_0 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_0
)
test_mnist_rotated_0 = data_utils.Subset(test_mnist_rotated_0, test_indices)



train_mnist_rotated_40 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_40
)
train_mnist_rotated_40 = data_utils.Subset(train_mnist_rotated_40, train_indices)



test_mnist_rotated_40 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_40
)
test_mnist_rotated_40 = data_utils.Subset(test_mnist_rotated_40, test_indices)



train_mnist_rotated_80 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_80
)
train_mnist_rotated_80 = data_utils.Subset(train_mnist_rotated_80, train_indices)



test_mnist_rotated_80 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_80
)
test_mnist_rotated_80 = data_utils.Subset(test_mnist_rotated_80, test_indices)



train_mnist_rotated_120 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_120
)
train_mnist_rotated_120 = data_utils.Subset(train_mnist_rotated_120, train_indices)



test_mnist_rotated_120 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_120
)
test_mnist_rotated_120 = data_utils.Subset(test_mnist_rotated_120, test_indices)




train_mnist_rotated_160 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_160
)
train_mnist_rotated_160 = data_utils.Subset(train_mnist_rotated_160, train_indices)




test_mnist_rotated_160 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_160
)
test_mnist_rotated_160 = data_utils.Subset(test_mnist_rotated_160, test_indices)

print(train_indices)

print(test_indices)

# Alternatively, task labels can also be a list (or tensor)
# containing the task label of each pattern
from avalanche.benchmarks.utils import make_classification_dataset, AvalancheDataset


train_mnist_rotated_0_task1 = make_classification_dataset(train_mnist_rotated_0, task_labels=1)
test_mnist_rotated_0_task1 = make_classification_dataset(test_mnist_rotated_0, task_labels=1)


train_mnist_rotated_40_task1 = make_classification_dataset(train_mnist_rotated_40, task_labels=2)

test_mnist_rotated_40_task1 = make_classification_dataset(test_mnist_rotated_40, task_labels=2)


train_mnist_rotated_80_task1 = make_classification_dataset(train_mnist_rotated_80, task_labels=3)

test_mnist_rotated_80_task1 = make_classification_dataset(test_mnist_rotated_80, task_labels=3)


train_mnist_rotated_120_task1 = make_classification_dataset(train_mnist_rotated_120, task_labels=4)

test_mnist_rotated_120_task1 = make_classification_dataset(test_mnist_rotated_120, task_labels=4)


train_mnist_rotated_160_task1 = make_classification_dataset(train_mnist_rotated_160, task_labels=5)

test_mnist_rotated_160_task1 = make_classification_dataset(test_mnist_rotated_160, task_labels=5)




scenario_custom_task_labels_rotated_mnist = dataset_benchmark(
    [train_mnist_rotated_0_task1, train_mnist_rotated_40_task1, train_mnist_rotated_80_task1, train_mnist_rotated_120_task1, train_mnist_rotated_160_task1],
    [test_mnist_rotated_0_task1, test_mnist_rotated_40_task1,test_mnist_rotated_80_task1, test_mnist_rotated_120_task1, test_mnist_rotated_160_task1]
)

# recovering the train and test streams
train_stream = scenario_custom_task_labels_rotated_mnist.train_stream
test_stream = scenario_custom_task_labels_rotated_mnist.test_stream

print('--- Original datasets:')
print(scenario_custom_task_labels_rotated_mnist.original_train_dataset)
print(scenario_custom_task_labels_rotated_mnist.original_test_dataset)


print('--- Task labels:')
print(scenario_custom_task_labels_rotated_mnist.task_labels)

# train and test streams
print('--- Streams:')
print(scenario_custom_task_labels_rotated_mnist.train_stream)
print(scenario_custom_task_labels_rotated_mnist.test_stream)



# we get the first experience
experience = train_stream[0]

# task label and dataset are the main attributes
t_label = experience.task_label
dataset = experience.dataset

# but you can recover additional info
experience.current_experience
experience.classes_in_this_experience
experience.classes_seen_so_far
experience.previous_classes
experience.future_classes
experience.origin_stream
experience.benchmark

# As always, we can iterate over it normally or with a pytorch
# data loader.
# For instance, we can use tqdm to add a progress bar.
from tqdm import tqdm
for i, data in enumerate(tqdm(dataset)):
  pass
print("\nNumber of examples:", i + 1)
print("Task Label:", t_label)

# iterating over the train stream
i=5
for experience in train_stream:
    print("Start of task ", experience.task_label)
    print('Classes in this task:', experience.classes_in_this_experience)

    # The current Pytorch training set can be easily recovered through the
    # experience
    current_training_set = experience.dataset
    # ...as well as the task_label
    print('Task {}'.format(experience.task_label))
    print('This task contains', len(current_training_set), 'training examples')

    # we can recover the corresponding test experience in the test stream
    current_test_set = test_stream[experience.current_experience].dataset
    print('This task contains', len(current_test_set), 'test examples')

    #the tasks do not contain specific labels. Hence this is "Domain Incremental Scenario"

"""## Strategies

### Strategy 1: SI
"""

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from avalanche.training.supervised import SynapticIntelligence, EWC

from torch.optim.adam import Adam

model_domain_incre_si = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=400, hidden_layers=2, drop_rate=0.15)

optimizer = Adam(model_domain_incre_si.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_si = SynapticIntelligence(
    model_domain_incre_si, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, si_lambda = 5
)


"""#### Training"""

# TRAINING LOOP
print('Starting experiment...')
results_rotated_mnist_si = []
for experience in train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_si.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_rotated_mnist_si.append(cl_strategy_si.eval(test_stream))

"""#### Saving the Model"""

torch.save(model_domain_incre_si.state_dict(), 'model_domain_incre_si_final.pth')

"""#### Evaluation"""

print(results_rotated_mnist_si)

##### FINAL Evaluation Code
# Only determine the correct key
counter = 0
for dict in results_rotated_mnist_si:
  
  for key,value in dict.items():
    
    
    #task 1
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_rotated_mnist.loc["SI","Accuracy - Task 0"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_rotated_mnist.loc["SI","Loss - Task 0"] = value
    # task 2
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_rotated_mnist.loc["SI","Accuracy - Task 1"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_rotated_mnist.loc["SI","Loss - Task 1"] = value
    # task 3
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_rotated_mnist.loc["SI","Accuracy - Task 2"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_rotated_mnist.loc["SI","Loss - Task 2"] = value
      
    #task 4
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task004/Exp003":
      performance_rotated_mnist.loc["SI","Accuracy - Task 3"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task004/Exp003":
      performance_rotated_mnist.loc["SI","Loss - Task 3"] = value
    # task 5
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task005/Exp004":
      performance_rotated_mnist.loc["SI","Accuracy - Task 4"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task005/Exp004":
      performance_rotated_mnist.loc["SI","Loss - Task 4"] = value
    # Overall accuracy
    if key == "Top1_Acc_Stream/eval_phase/test_stream/Task005":
      performance_rotated_mnist.loc["SI","Accuracy"] = value
      
    if key == "Loss_Stream/eval_phase/test_stream/Task005":
      performance_rotated_mnist.loc["SI","Loss"] = value
  counter += 1
print(performance_rotated_mnist)




"""### Strategy 2: EWC"""

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from torch.optim.adam import Adam
model_domain_incre_ewc = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=400, hidden_layers=2, drop_rate=0.15)

optimizer = Adam(model_domain_incre_ewc.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_ewc = EWC(
    model_domain_incre_ewc, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""#### Training"""

# TRAINING LOOP
print('Starting experiment...')
results_rotated_mnist_ewc = []
for experience in train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_ewc.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_rotated_mnist_ewc.append(cl_strategy_ewc.eval(test_stream))

"""#### Saving the Model"""

torch.save(model_domain_incre_ewc.state_dict(), 'model_domain_incre_ewc_final.pth')

"""#### Evaluation"""

print (results_rotated_mnist_ewc)

##### FINAL Evaluation Code
# Only determine the correct key
counter = 0
for dict in results_rotated_mnist_ewc:
  
  for key,value in dict.items():
    
    
    #task 1
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_rotated_mnist.loc["EWC","Accuracy - Task 0"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_rotated_mnist.loc["EWC","Loss - Task 0"] = value
    # task 2
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_rotated_mnist.loc["EWC","Accuracy - Task 1"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_rotated_mnist.loc["EWC","Loss - Task 1"] = value
    # task 3
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_rotated_mnist.loc["EWC","Accuracy - Task 2"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_rotated_mnist.loc["EWC","Loss - Task 2"] = value
      
    #task 4
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task004/Exp003":
      performance_rotated_mnist.loc["EWC","Accuracy - Task 3"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task004/Exp003":
      performance_rotated_mnist.loc["EWC","Loss - Task 3"] = value
    # task 5
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task005/Exp004":
      performance_rotated_mnist.loc["EWC","Accuracy - Task 4"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task005/Exp004":
      performance_rotated_mnist.loc["EWC","Loss - Task 4"] = value
    # Overall accuracy
    if key == "Top1_Acc_Stream/eval_phase/test_stream/Task005":
      performance_rotated_mnist.loc["EWC","Accuracy"] = value
      
    if key == "Loss_Stream/eval_phase/test_stream/Task005":
      performance_rotated_mnist.loc["EWC","Loss"] = value
  counter += 1
print(performance_rotated_mnist)


"""### Strategy 3: Online EWC"""

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from avalanche.training.supervised import EWC

class OnlineEWC(EWC):
    def before_training_exp(self, strategy, **kwargs):
        self.model.freeze()
        self.estimated_importance = {}  # Dictionary to store parameter importance estimates
        self.online_fisher = {}  # Dictionary to store online Fisher information estimates

    def before_backward(self, strategy, **kwargs):
        # Compute gradients for each parameter and update online Fisher information
        gradients = self.model.grad()
        for name, gradient in gradients.items():
            self.online_fisher[name] += gradient ** 2  # Update online Fisher information

    def after_training_epoch(self, strategy, **kwargs):
        # Update parameter importance based on online Fisher information
        for name, parameter in self.model.named_parameters():
            self.estimated_importance[name] = self.online_fisher[name]  # Update parameter importance

    def penalty(self, strategy, **kwargs):
        # Compute penalty based on parameter importance
        penalty = 0.0
        for name, parameter in self.model.named_parameters():
            importance = self.estimated_importance[name]
            penalty += (parameter - self.initial_parameters[name]) ** 2 * importance
        return penalty

from torch.optim.adam import Adam
model_domain_incre_online_ewc = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=400, hidden_layers=2, drop_rate=0.15)

optimizer = Adam(model_domain_incre_online_ewc.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_online_ewc = OnlineEWC(
    model_domain_incre_online_ewc, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""#### Training"""

# TRAINING LOOP
print('Starting experiment...')
results_rotated_mnist_online_ewc = []
for experience in train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_online_ewc.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_rotated_mnist_online_ewc.append(cl_strategy_online_ewc.eval(test_stream))

"""#### Saving the Model"""

torch.save(model_domain_incre_online_ewc.state_dict(), 'model_domain_incre_online_ewc_final.pth')

"""#### Evaluation"""

print(results_rotated_mnist_online_ewc)

##### FINAL Evaluation Code
# Only determine the correct key
counter = 0
for dict in results_rotated_mnist_online_ewc:
  
  for key,value in dict.items():
    
    
    #task 1
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_rotated_mnist.loc["Online EWC","Accuracy - Task 0"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_rotated_mnist.loc["Online EWC","Loss - Task 0"] = value
    # task 2
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_rotated_mnist.loc["Online EWC","Accuracy - Task 1"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_rotated_mnist.loc["Online EWC","Loss - Task 1"] = value
    # task 3
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_rotated_mnist.loc["Online EWC","Accuracy - Task 2"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_rotated_mnist.loc["Online EWC","Loss - Task 2"] = value
      
    #task 4
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task004/Exp003":
      performance_rotated_mnist.loc["Online EWC","Accuracy - Task 3"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task004/Exp003":
      performance_rotated_mnist.loc["Online EWC","Loss - Task 3"] = value
    # task 5
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task005/Exp004":
      performance_rotated_mnist.loc["Online EWC","Accuracy - Task 4"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task005/Exp004":
      performance_rotated_mnist.loc["Online EWC","Loss - Task 4"] = value
    # Overall accuracy
    if key == "Top1_Acc_Stream/eval_phase/test_stream/Task005":
      performance_rotated_mnist.loc["Online EWC","Accuracy"] = value
      
    if key == "Loss_Stream/eval_phase/test_stream/Task005":
      performance_rotated_mnist.loc["Online EWC","Loss"] = value
  counter += 1
print(performance_rotated_mnist)
  

performance_rotated_mnist.to_csv("continual_learning_performance_RotatedMNIST_final.csv")


"""## Scenario 2: Task Incremental - MNIST Fellowship"""

from avalanche.benchmarks.datasets import MNIST, KMNIST, FashionMNIST
from avalanche.benchmarks.generators import dataset_benchmark
import pandas as pd

performance_mnist_fellowship = pd.DataFrame(index=["SI","EWC","Online EWC"], columns=["Loss - Task 0","Loss - Task 1","Loss - Task 2","Accuracy - Task 0","Accuracy - Task 1","Accuracy - Task 2","Accuracy", "Loss"])

import torch.utils.data as data_utils
import random
train_indices = []
test_indices = []

train_indices = random.sample(range(0,60000),1200)
test_indices = random.sample(range(0,10000),200)

train_mnist = MNIST(
    './data/MNIST', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_mnist = data_utils.Subset(train_mnist, train_indices)



test_mnist = MNIST(
    './data/MNIST', train=False, download=True, transform=torchvision.transforms.ToTensor()
)
test_mnist = data_utils.Subset(test_mnist, test_indices)



train_kmnist = KMNIST(
    './data/KMNIST', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_kmnist = data_utils.Subset(train_kmnist, train_indices)


test_kmnist = KMNIST(
    './data/KMNIST', train=False, download=True, transform=torchvision.transforms.ToTensor()
)
test_kmnist = data_utils.Subset(test_kmnist, test_indices)



train_fashion_mnist = FashionMNIST(
    './data/FashionMNIST', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
train_fashion_mnist = data_utils.Subset(train_fashion_mnist, train_indices)


test_fashion_mnist = FashionMNIST(
    './data/FashionMNIST', train=False, download=True, transform=torchvision.transforms.ToTensor()
)
test_fashion_mnist = data_utils.Subset(test_fashion_mnist, test_indices)

print("len(train_mnist",len(train_mnist))

print('len(train_kmnist)',len(train_kmnist))

print('len(train_kmnist)',len(train_fashion_mnist))

print('')

print('len(test_mnist)',len(test_mnist))

print('len(test_kmnist)',len(test_kmnist))

print('len(test_fashion_mnist)',len(test_fashion_mnist))

generic_scenario = dataset_benchmark(
    [train_mnist, train_kmnist, train_fashion_mnist],
    [test_mnist, test_kmnist, test_fashion_mnist]
)

# Alternatively, task labels can also be a list (or tensor)
# containing the task label of each pattern
from avalanche.benchmarks.utils import make_classification_dataset

train_MNIST_task1 = make_classification_dataset(train_mnist, task_labels=1)
test_MNIST_task1 = make_classification_dataset(test_mnist, task_labels=1)

train_KMNIST_task1 = make_classification_dataset(train_kmnist, task_labels=2)
test_KMNIST_task1 = make_classification_dataset(test_kmnist, task_labels=2)

train_fashion_mnist_task1 = make_classification_dataset(train_fashion_mnist, task_labels=3)
test_fashion_mnist_task1 = make_classification_dataset(test_fashion_mnist, task_labels=3)

scenario_custom_task_labels = dataset_benchmark(
    [train_MNIST_task1, train_KMNIST_task1,train_fashion_mnist_task1],
    [test_MNIST_task1, test_KMNIST_task1,test_fashion_mnist_task1]
)

print('Without custom task labels:',
      generic_scenario.train_stream[1].task_label)

print('With custom task labels:',
      scenario_custom_task_labels.train_stream[2].task_label)

train_stream_task_incremental = scenario_custom_task_labels.train_stream
test_stream_task_incremental = scenario_custom_task_labels.test_stream

"""## Strategies

### Strategy 1: SI
"""

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from avalanche.training.supervised import SynapticIntelligence, EWC

from torch.optim.adam import Adam
model_task_incre_si = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=1000, hidden_layers=2)#, drop_rate=0.15)

optimizer = Adam(model_task_incre_si.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_si_task_incremental = SynapticIntelligence(
    model_task_incre_si, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, si_lambda = 5
)

"""#### Training"""

# TRAINING LOOP
print('Starting experiment...')
results_mnist_fellowship_si = []
for experience in train_stream_task_incremental:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_si_task_incremental.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_mnist_fellowship_si.append(cl_strategy_si_task_incremental.eval(test_stream_task_incremental))

"""#### Saving the Model"""

torch.save(model_task_incre_si.state_dict(), 'model_task_incre_si.pth')

"""#### Comments

After careful analysis of the numbers generated during training, the following can be concluded:


*   Each task is trained sequentially i.e. task 0 is trained first, then there is evaluation for all task (in that evaluation, only task 1 performs well). Then the next task is trained. And again all the tasks are evaluated. In the second case, after task 0 and task 1 are trained, both the tasks should perform well in evaluation, but only task 2 performs well
*   Thus, there is catastrophic forgetting here in synaptic intelligence
*   In the end, task 3 performs very well, because it was trained recently on the network

#### Evaluation
"""

print(results_mnist_fellowship_si)

##### FINAL Evaluation Code
# Only determine the correct key
counter = 0
for dict in results_mnist_fellowship_si:
  
  for key,value in dict.items():
    
    
    #task 1
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_mnist_fellowship.loc["SI","Accuracy - Task 0"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_mnist_fellowship.loc["SI","Loss - Task 0"] = value
    # task 2
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_mnist_fellowship.loc["SI","Accuracy - Task 1"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_mnist_fellowship.loc["SI","Loss - Task 1"] = value
    # task 3
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_mnist_fellowship.loc["SI","Accuracy - Task 2"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_mnist_fellowship.loc["SI","Loss - Task 2"] = value
      
      
    # Overall accuracy
    if key == "Top1_Acc_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["SI","Accuracy"] = value
      
    if key == "Loss_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["SI","Loss"] = value
  counter += 1
print(performance_mnist_fellowship)
  


"""### Strategy 2: EWC"""

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from torch.optim.adam import Adam
model = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=1000, hidden_layers=2)#, drop_rate=0.15)

optimizer = Adam(model.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_ewc_task_incremental = EWC(
    model, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""#### Training

"""

# TRAINING LOOP
print('Starting experiment...')
results_mnist_fellowship_ewc = []
for experience in train_stream_task_incremental:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_ewc_task_incremental.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_mnist_fellowship_ewc.append(cl_strategy_ewc_task_incremental.eval(test_stream_task_incremental))

"""#### Saving the Model"""

torch.save(model.state_dict(), 'model_task_incre_ewc.pth')

"""#### Evaluation"""

print(results_mnist_fellowship_ewc)

##### FINAL Evaluation Code
# Only determine the correct key
counter = 0
for dict in results_mnist_fellowship_ewc:
  
  for key,value in dict.items():
    
    
    #task 1
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_mnist_fellowship.loc["EWC","Accuracy - Task 0"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_mnist_fellowship.loc["EWC","Loss - Task 0"] = value
    # task 2
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_mnist_fellowship.loc["EWC","Accuracy - Task 1"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_mnist_fellowship.loc["EWC","Loss - Task 1"] = value
    # task 3
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_mnist_fellowship.loc["EWC","Accuracy - Task 2"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_mnist_fellowship.loc["EWC","Loss - Task 2"] = value
      
    
    # Overall accuracy
    if key == "Top1_Acc_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["EWC","Accuracy"] = value
      
    if key == "Loss_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["EWC","Loss"] = value
  counter += 1
print(performance_mnist_fellowship)
  



"""### Strategy 3: Online EWC"""

from avalanche.training.supervised import EWC

class OnlineEWC(EWC):
    def before_training_exp(self, strategy, **kwargs):
        self.model.freeze()
        self.estimated_importance = {}  # Dictionary to store parameter importance estimates
        self.online_fisher = {}  # Dictionary to store online Fisher information estimates

    def before_backward(self, strategy, **kwargs):
        # Compute gradients for each parameter and update online Fisher information
        gradients = self.model.grad()
        for name, gradient in gradients.items():
            self.online_fisher[name] += gradient ** 2  # Update online Fisher information

    def after_training_epoch(self, strategy, **kwargs):
        # Update parameter importance based on online Fisher information
        for name, parameter in self.model.named_parameters():
            self.estimated_importance[name] = self.online_fisher[name]  # Update parameter importance

    def penalty(self, strategy, **kwargs):
        # Compute penalty based on parameter importance
        penalty = 0.0
        for name, parameter in self.model.named_parameters():
            importance = self.estimated_importance[name]
            penalty += (parameter - self.initial_parameters[name]) ** 2 * importance
        return penalty

from torch.optim.adam import Adam
from avalanche.models import SimpleMLP
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!
model_task_incre_online_ewc = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=1000, hidden_layers=2)#, drop_rate = 0.15)

optimizer = Adam(model_task_incre_online_ewc.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_online_ewc = OnlineEWC(
    model_task_incre_online_ewc, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""#### Training"""

# TRAINING LOOP
print('Starting experiment...')
results_mnist_fellowship_online_ewc = []
for experience in train_stream_task_incremental:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_online_ewc.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_mnist_fellowship_online_ewc.append(cl_strategy_online_ewc.eval(train_stream_task_incremental))

"""#### Saving the Model"""

torch.save(model_task_incre_online_ewc.state_dict(), 'model_task_incre_online_ewc.pth')

"""#### Evaluation"""

print(results_mnist_fellowship_online_ewc)

##### FINAL Evaluation Code
# Only determine the correct key
counter = 0
for dict_1 in results_mnist_fellowship_online_ewc:
  
  for key1,value1 in dict_1.items():
    
    
    #task 1
    if key1 == "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_mnist_fellowship.loc["Online EWC","Accuracy - Task 0"] = value1
      
    if key1 == "Loss_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_mnist_fellowship.loc["Online EWC","Loss - Task 0"] = value1
    # task 2
    if key1 == "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_mnist_fellowship.loc["Online EWC","Accuracy - Task 1"] = value1
      
    if key1 == "Loss_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_mnist_fellowship.loc["Online EWC","Loss - Task 1"] = value1
    # task 3
    if key1 == "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_mnist_fellowship.loc["Online EWC","Accuracy - Task 2"] = value1
      
    if key1 == "Loss_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_mnist_fellowship.loc["Online EWC","Loss - Task 2"] = value1
      

    # Overall accuracy
    if key1 == "Top1_Acc_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["Online EWC","Accuracy"] = value1
      
    if key1 == "Loss_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["Online EWC","Loss"] = value1
  counter += 1
print(performance_mnist_fellowship)
  

"""# Saving Performance to CSV File"""

performance_mnist_fellowship.to_csv("continual_learning_performance_MNIST_fellowship.csv")

