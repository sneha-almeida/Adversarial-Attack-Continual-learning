# -*- coding: utf-8 -*-
"""continual_learning_label_flipping.ipynb
"""


import torch
import torchvision

"""#

## Scenario 1: Domain Incremental - RotatedMNIST
"""

from avalanche.benchmarks.classic import RotatedMNIST
import pandas as pd

performance_rotated_mnist = pd.DataFrame(index=["SI","EWC","Online EWC"], columns=["Loss - Task 0","Loss - Task 1","Loss - Task 2","Loss - Task 3","Loss - Task 4","Accuracy - Task 0","Accuracy - Task 1","Accuracy - Task 2","Accuracy - Task 3","Accuracy - Task 4","Accuracy", "Loss"])



import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from avalanche.benchmarks import dataset_benchmark

rotation_angle_0 = (0.0,0.1)  # Specify the desired rotation angle in degrees
rotation_angle_40 = (40.0,40.1)
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
test_mnist_rotated_0 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_0
)


train_mnist_rotated_40 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_40
)
test_mnist_rotated_40 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_40
)


train_mnist_rotated_80 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_80
)
test_mnist_rotated_80 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_80
)


train_mnist_rotated_120 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_120
)
test_mnist_rotated_120 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_120
)


train_mnist_rotated_160 = MNIST(
    './data/MNIST', train=True, download=True, transform=rotation_transform_160
)
test_mnist_rotated_160 = MNIST(
    './data/MNIST', train=False, download=True, transform=rotation_transform_160
)

"""#### Change Labels of the First Task"""

#################### FIRST TASK IS MNIST ROTATED BY 0 DEGRESS #####################
import random
from tqdm import tqdm

poisoned_data = []
poisoned_labels = []
poisoned_samples_list = []
percentage_labels_to_be_changed = 0.1

labels_to_be_changed = len(train_mnist_rotated_0)*percentage_labels_to_be_changed
labels_to_be_changed_list = random.sample(range(0, len(train_mnist_rotated_0)), int(labels_to_be_changed))
for i, data in enumerate(tqdm(train_mnist_rotated_0)):
  if i in labels_to_be_changed_list:
    poisoned_label = (data[1] + random.randint(1, 9)) % 10
    poisoned_sample = [data[0], poisoned_label]
    poisoned_data.append(data[0])
    poisoned_labels.append(poisoned_label)
    poisoned_samples_list.append(poisoned_sample)

#print("length of poisoned samples",len(poisoned_samples_list))
#print("poisoned labels",poisoned_labels)
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.utils.data import ConcatDataset

dataset = ClassificationDataset([poisoned_samples_list])



"""#### Inject changed labels into other tasks"""

task_labels_train = [1] * 66000

task_labels_test = [1] * 10000

poisoned_train_mnist_rotated_40 = ClassificationDataset([train_mnist_rotated_40,poisoned_samples_list])
poisoned_train_mnist_rotated_40.targets_task_labels = task_labels_train
test_mnist_rotated_40.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_80 = ClassificationDataset([train_mnist_rotated_80,poisoned_samples_list])
poisoned_train_mnist_rotated_80.targets_task_labels = task_labels_train
test_mnist_rotated_80.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_120 = ClassificationDataset([train_mnist_rotated_120,poisoned_samples_list])
poisoned_train_mnist_rotated_120.targets_task_labels = task_labels_train
test_mnist_rotated_120.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_160 = ClassificationDataset([train_mnist_rotated_160,poisoned_samples_list])
poisoned_train_mnist_rotated_160.targets_task_labels = task_labels_train
test_mnist_rotated_160.targets_task_labels = task_labels_test

from avalanche.benchmarks.utils.data_attribute import DataAttribute
#extract labels from datasets (we will put those in ClassificationDataset.targets)
poisoned_train_mnist_rotated_40_labels = []
for i, data in enumerate(tqdm(poisoned_train_mnist_rotated_40)):
  label = (data[1])
  poisoned_train_mnist_rotated_40_labels.append(label)
print(len(poisoned_train_mnist_rotated_40_labels))

poisoned_train_mnist_rotated_40_labels_final = DataAttribute(poisoned_train_mnist_rotated_40_labels)


poisoned_train_mnist_rotated_80_labels = []
for i, data in enumerate(tqdm(poisoned_train_mnist_rotated_80)):
  label = (data[1])
  poisoned_train_mnist_rotated_80_labels.append(label)
print(len(poisoned_train_mnist_rotated_80_labels))

poisoned_train_mnist_rotated_80_labels_final = DataAttribute(poisoned_train_mnist_rotated_80_labels)


poisoned_train_mnist_rotated_120_labels = []
for i, data in enumerate(tqdm(poisoned_train_mnist_rotated_120)):
  label = (data[1])
  poisoned_train_mnist_rotated_120_labels.append(label)
print(len(poisoned_train_mnist_rotated_120_labels))

poisoned_train_mnist_rotated_120_labels_final = DataAttribute(poisoned_train_mnist_rotated_120_labels)


poisoned_train_mnist_rotated_160_labels = []
for i, data in enumerate(tqdm(poisoned_train_mnist_rotated_160)):
  label = (data[1])
  poisoned_train_mnist_rotated_160_labels.append(label)
print(len(poisoned_train_mnist_rotated_160_labels))

poisoned_train_mnist_rotated_160_labels_final = DataAttribute(poisoned_train_mnist_rotated_160_labels)


# Alternatively, task labels can also be a list (or tensor)
# containing the task label of each pattern
from avalanche.benchmarks.utils import make_classification_dataset, AvalancheDataset


train_mnist_rotated_0_task1 = make_classification_dataset(train_mnist_rotated_0, task_labels=1)
test_mnist_rotated_0_task1 = make_classification_dataset(test_mnist_rotated_0, task_labels=1)


train_mnist_rotated_40_task1 = make_classification_dataset(poisoned_train_mnist_rotated_40, task_labels=2)
train_mnist_rotated_40_task1 = AvalancheDataset(train_mnist_rotated_40_task1)
train_mnist_rotated_40_task1.targets = poisoned_train_mnist_rotated_40_labels_final
test_mnist_rotated_40_task1 = make_classification_dataset(test_mnist_rotated_40, task_labels=2)


train_mnist_rotated_80_task1 = make_classification_dataset(poisoned_train_mnist_rotated_80, task_labels=3)
train_mnist_rotated_80_task1 = AvalancheDataset(train_mnist_rotated_80_task1)
train_mnist_rotated_80_task1.targets = poisoned_train_mnist_rotated_80_labels_final
test_mnist_rotated_80_task1 = make_classification_dataset(test_mnist_rotated_80, task_labels=3)


train_mnist_rotated_120_task1 = make_classification_dataset(poisoned_train_mnist_rotated_120, task_labels=4)
train_mnist_rotated_120_task1 = AvalancheDataset(train_mnist_rotated_120_task1)
train_mnist_rotated_120_task1.targets = poisoned_train_mnist_rotated_120_labels_final
test_mnist_rotated_120_task1 = make_classification_dataset(test_mnist_rotated_120, task_labels=4)


train_mnist_rotated_160_task1 = make_classification_dataset(poisoned_train_mnist_rotated_160, task_labels=5)
train_mnist_rotated_160_task1 = AvalancheDataset(train_mnist_rotated_160_task1)
train_mnist_rotated_160_task1.targets = poisoned_train_mnist_rotated_160_labels_final
test_mnist_rotated_160_task1 = make_classification_dataset(test_mnist_rotated_160, task_labels=5)




scenario_custom_task_labels_rotated_mnist = dataset_benchmark(
    [train_mnist_rotated_0_task1, train_mnist_rotated_40_task1, train_mnist_rotated_80_task1, train_mnist_rotated_120_task1, train_mnist_rotated_160_task1],
    [test_mnist_rotated_0_task1, test_mnist_rotated_40_task1,test_mnist_rotated_80_task1, test_mnist_rotated_120_task1, test_mnist_rotated_160_task1]
)


# recovering the train and test streams
train_stream = scenario_custom_task_labels_rotated_mnist.train_stream
test_stream = scenario_custom_task_labels_rotated_mnist.test_stream

counter = 0
# Iterate over the train_stream and print task labels
for task_info in scenario_custom_task_labels_rotated_mnist.train_stream:
    task_label = task_info.task_label
    
    
    counter += counter

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
print(experience.classes_in_this_experience)
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


import numpy

"""#### Label Flipping"""

from avalanche.benchmarks.utils import AvalancheConcatDataset

experience0 = train_stream[0]
dataset_0 = experience0.dataset
print(len(list(dataset_0)))

experience1 = train_stream[1]
dataset_1 = experience1.dataset
print(len(list(dataset_1)))

experience2 = train_stream[2]
dataset_2 = experience2.dataset
print(len(list(dataset_2)))

experience3 = train_stream[3]
dataset_3 = experience3.dataset
print(len(list(dataset_3)))

experience4 = train_stream[4]
dataset_4 = experience4.dataset
print(len(list(dataset_4)))
## Here we have to take samples from task 0, change their label and inject them into other tasks

print(dataset_1[0])

"""##### Subset of data points to be injected from target task"""

import random
percentage_labels_to_be_changed = 0.1

original_data = []
original_labels = []
poisoned_data = []
poisoned_labels = []
poisoned_samples_list = []
labels_to_be_changed = len(dataset_0)*percentage_labels_to_be_changed
print("len(dataset_0)",len(dataset_0))
#print("labels_to_be_changed",labels_to_be_changed)
labels_to_be_changed_list = random.sample(range(0, len(dataset_0)), int(labels_to_be_changed))
print("how many labels to be changed: ", len(labels_to_be_changed_list))
for i, data in enumerate(tqdm(dataset_0)):
    #data is a list of 3 items
    #1. Data point
    #2. Data Label
    #3. Task Label
    original_data.append(data[0])
    original_labels.append(data[1])
    #Flip the label of the datapoint if it has been selected randomly
    if i in labels_to_be_changed_list:
        #randomly select a new label from the 0th epcerience because we only change the labels of the first task and inject those into other tasks
        #DON"T MAKE CHANGES TO THE DATA IN EXISTING EXPERIENCE
        poisoned_label = (data[1] + random.randint(1, len(experience0.classes_in_this_experience)-1))% len(experience0.classes_in_this_experience)
        #add data[0], poisoned_label and data[2] to all other experiences
        poisoned_sample = [data[0], poisoned_label, data[2]]
        poisoned_data.append(data[0])
        poisoned_labels.append(poisoned_label)
        poisoned_samples_list.append(poisoned_sample)
print("len of poisoned labels: ", len(poisoned_labels))
print("poisoned labels: ", poisoned_labels)


#print("length of poisoned samples",len(poisoned_samples_list))

from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.utils.data import ConcatDataset

dataset = ClassificationDataset([poisoned_samples_list])

"""##### Injecting in task 1"""

concatenated_dataset = ConcatDataset([dataset_1, dataset])

train_stream[1].dataset = concatenated_dataset

"""##### injecting in task 2"""

concatenated_dataset = ConcatDataset([dataset_2, dataset])
dataset = ClassificationDataset([poisoned_samples_list])
concatenated_dataset = ConcatDataset([dataset_2, dataset])
train_stream[2].dataset = concatenated_dataset

"""##### injecting in task 3"""

concatenated_dataset = ConcatDataset([dataset_3, dataset])
dataset = ClassificationDataset([poisoned_samples_list])
concatenated_dataset = ConcatDataset([dataset_3, dataset])
train_stream[3].dataset = concatenated_dataset

"""injecting in task 4"""

concatenated_dataset = ConcatDataset([dataset_4, dataset])
dataset = ClassificationDataset([poisoned_samples_list])
concatenated_dataset = ConcatDataset([dataset_4, dataset])
print(len(list(concatenated_dataset)))
train_stream[4].dataset = concatenated_dataset
print(len(list(train_stream[4].dataset)))

experience0 = train_stream[0]
dataset_0 = experience0.dataset
print(len(list(dataset_0)))

experience1 = train_stream[1]
dataset_1 = experience1.dataset
print(len(list(dataset_1)))

experience2 = train_stream[2]
dataset_2 = experience2.dataset
print(len(list(dataset_2)))

experience3 = train_stream[3]
dataset_3 = experience3.dataset
print(len(list(dataset_3)))

experience4 = train_stream[4]
dataset_4 = experience4.dataset
print(len(list(dataset_4)))

# iterating over the train stream
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

performance_mnist_fellowship = pd.DataFrame(index=["SI","EWC","Online EWC"], columns=["Loss - Task 0","Loss - Task 1","Loss - Task 2","Accuracy - Task 0","Accuracy - Task 1","Accuracy - Task 2","Accuracy", "Loss"])

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
counter = 0
for experience in train_stream:
    
    
    
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_si.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_rotated_mnist_si.append(cl_strategy_si.eval(test_stream))

results_rotated_mnist_si_label_flipping = []
results_rotated_mnist_si.append(cl_strategy_si.eval(test_stream))

torch.save(model_domain_incre_si.state_dict(), 'model_domain_incre_si.pth')

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
  

import matplotlib.pyplot as plt
import pandas as pd


# Extract the keys and values from the data dictionary
keys = list(results_rotated_mnist_si[0].keys())
values = [list(d.values()) for d in results_rotated_mnist_si]

# Create a Pandas DataFrame
df = pd.DataFrame(values, columns=keys)

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

torch.save(model_domain_incre_ewc.state_dict(), 'model_domain_incre_ewc.pth')

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
  



import matplotlib.pyplot as plt
import pandas as pd


# Extract the keys and values from the data dictionary
keys = list(results_rotated_mnist_ewc[0].keys())
values = [list(d.values()) for d in results_rotated_mnist_ewc]

# Create a Pandas DataFrame
df = pd.DataFrame(values, columns=keys)


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

torch.save(model_domain_incre_online_ewc.state_dict(), 'model_domain_incre_online_ewc_label_flipping.pth')

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
  #print(dict)

performance_rotated_mnist.to_csv("continual_learning_performance_label_flipping_rotated_mnist.csv")

import matplotlib.pyplot as plt
import pandas as pd


# Extract the keys and values from the data dictionary
keys = list(results_rotated_mnist_online_ewc[0].keys())
values = [list(d.values()) for d in results_rotated_mnist_online_ewc]

# Create a Pandas DataFrame
df = pd.DataFrame(values, columns=keys)


"""## Scenario 2: Task Incremental - MNIST Fellowship"""

import pandas as pd

from avalanche.benchmarks.datasets import MNIST, KMNIST, FashionMNIST
from avalanche.benchmarks.generators import dataset_benchmark

performance_mnist_fellowship = pd.DataFrame(index=["SI","EWC","Online EWC"], columns=["Loss - Task 0","Loss - Task 1","Loss - Task 2","Accuracy - Task 0","Accuracy - Task 1","Accuracy - Task 2","Accuracy", "Loss"])

train_mnist = MNIST(
    './data/MNIST', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_mnist = MNIST(
    './data/MNIST', train=False, download=True, transform=torchvision.transforms.ToTensor()
)

train_kmnist = KMNIST(
    './data/KMNIST', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_kmnist = KMNIST(
    './data/KMNIST', train=False, download=True, transform=torchvision.transforms.ToTensor()
)

train_fashion_mnist = FashionMNIST(
    './data/FashionMNIST', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_fashion_mnist = FashionMNIST(
    './data/FashionMNIST', train=False, download=True, transform=torchvision.transforms.ToTensor()
)

"""perform label flipping attack here itself

#### Change labels of the first task
"""

import random
from tqdm import tqdm

poisoned_data = []
poisoned_labels = []
poisoned_samples_list = []
percentage_labels_to_be_changed = 0.1

labels_to_be_changed = len(train_mnist)*percentage_labels_to_be_changed
labels_to_be_changed_list = random.sample(range(0, len(train_mnist)), int(labels_to_be_changed))
for i, data in enumerate(tqdm(train_mnist)):
  if i in labels_to_be_changed_list:
    poisoned_label = (data[1] + random.randint(1, 9))% 10
    poisoned_sample = [data[0], poisoned_label]
    poisoned_data.append(data[0])
    poisoned_labels.append(poisoned_label)
    poisoned_samples_list.append(poisoned_sample)

#print("length of poisoned samples",len(poisoned_samples_list))
#print("poisoned labels",poisoned_labels)
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.utils.data import ConcatDataset

dataset = ClassificationDataset([poisoned_samples_list])

"""#### Inject changed labels into other tasks"""

task_labels_train = [1] * 66000

task_labels_test = [1] * 10000

poisoned_kmnist = ClassificationDataset([train_kmnist,poisoned_samples_list])
poisoned_kmnist.targets_task_labels = task_labels_train
test_kmnist.targets_task_labels = task_labels_test


poisoned_fashion_mnist = ClassificationDataset([train_fashion_mnist,poisoned_samples_list])

from avalanche.benchmarks.utils.data_attribute import DataAttribute
#extract labels from datasets (we will put those in ClassificationDataset.targets)
posioned_kmnist_labels = []
for i, data in enumerate(tqdm(poisoned_kmnist)):
  label = data[1]
  posioned_kmnist_labels.append(label)
print(len(posioned_kmnist_labels))

posioned_kmnist_labels_final = DataAttribute(posioned_kmnist_labels)

from avalanche.benchmarks.utils.data_attribute import DataAttribute
#extract labels from datasets (we will put those in ClassificationDataset.targets)
poisoned_fashion_mnist_labels = []
for i, data in enumerate(tqdm(poisoned_fashion_mnist)):
  label = data[1]
  poisoned_fashion_mnist_labels.append(label)
print(len(poisoned_fashion_mnist_labels))

poisoned_fashion_mnist_labels_final = DataAttribute(poisoned_fashion_mnist_labels)

poisoned_kmnist.targets = posioned_kmnist_labels_final

poisoned_fashion_mnist.targets = poisoned_fashion_mnist_labels_final

#concatenated_dataset_kmnist = ConcatDataset([train_kmnist, dataset])
#print(type(concatenated_dataset_kmnist))
#concatenated_dataset_kmnist = ConcatDataset([train_fashion_mnist, dataset])



# Alternatively, task labels can also be a list (or tensor)
# containing the task label of each pattern
from avalanche.benchmarks.utils import make_classification_dataset, AvalancheDataset


train_MNIST_task1 = make_classification_dataset(train_mnist, task_labels=1)
print(dir(train_MNIST_task1))
test_MNIST_task1 = make_classification_dataset(test_mnist, task_labels=1)


train_KMNIST_task1 = make_classification_dataset(poisoned_kmnist, task_labels=2)
train_KMNIST_task1 = AvalancheDataset(train_KMNIST_task1)
train_KMNIST_task1.targets = posioned_kmnist_labels_final
test_KMNIST_task1 = make_classification_dataset(test_kmnist, task_labels=2)



train_fashion_mnist_task1 = make_classification_dataset(poisoned_fashion_mnist, task_labels=3)
train_fashion_mnist_task1 = AvalancheDataset(train_fashion_mnist_task1)
train_fashion_mnist_task1.targets = posioned_kmnist_labels_final
test_fashion_mnist_task1 = make_classification_dataset(test_fashion_mnist, task_labels=3)


scenario_custom_task_labels = dataset_benchmark(
    [train_MNIST_task1, train_KMNIST_task1,train_fashion_mnist_task1],
    [test_MNIST_task1, test_KMNIST_task1,test_fashion_mnist_task1]
)


train_stream_task_incremental = scenario_custom_task_labels.train_stream
test_stream_task_incremental = scenario_custom_task_labels.test_stream

"""#### Label Flipping"""

from avalanche.benchmarks.utils import AvalancheConcatDataset
from tqdm import tqdm

experience0 = train_stream_task_incremental[0]
dataset_0 = experience0.dataset

experience1 = train_stream_task_incremental[1]
dataset_1 = experience1.dataset

experience2 = train_stream_task_incremental[2]
dataset_2 = experience2.dataset



## Here we have to take samples from task 0, change their label and inject them into other tasks

"""##### Subset of data points to be injected from target task"""

import random
percentage_labels_to_be_changed = 0.1

original_data = []
original_labels = []
poisoned_data = []
poisoned_labels = []
poisoned_samples_list = []
labels_to_be_changed = len(dataset_0)*percentage_labels_to_be_changed


labels_to_be_changed_list = random.sample(range(0, len(dataset_0)), int(labels_to_be_changed))

for i, data in enumerate(tqdm(dataset_0)):
    #data is a list of 3 items
    #1. Data point
    #2. Data Label
    #3. Task Label
    original_data.append(data[0])
    original_labels.append(data[1])
    #Flip the label of the datapoint if it has been selected randomly
    if i in labels_to_be_changed_list:
        #randomly select a new label from the 0th epcerience because we only change the labels of the first task and inject those into other tasks
        #DON"T MAKE CHANGES TO THE DATA IN EXISTING EXPERIENCE
        
        
        poisoned_label = (data[1] + random.randint(1, len(experience0.classes_in_this_experience)-1))% len(experience0.classes_in_this_experience)

        #add data[0], poisoned_label and data[2] to all other experiences
        poisoned_sample = [data[0], poisoned_label, data[2]]
        poisoned_data.append(data[0])
        poisoned_labels.append(poisoned_label)
        poisoned_samples_list.append(poisoned_sample)


print("length of poisoned samples",len(poisoned_samples_list))

from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.utils.data import ConcatDataset

dataset = ClassificationDataset([poisoned_samples_list])


"""##### Injecting in task 1"""

concatenated_dataset = ConcatDataset([dataset_1, dataset])
#dataset = ClassificationDataset([poisoned_samples_list])
#concatenated_dataset = ConcatDataset([dataset_1, dataset])
#print(len(list(concatenated_dataset)))
train_stream_task_incremental[1].dataset = concatenated_dataset

"""##### injecting in task 2"""

concatenated_dataset = ConcatDataset([dataset_2, dataset])
dataset = ClassificationDataset([poisoned_samples_list])
concatenated_dataset = ConcatDataset([dataset_2, dataset])
train_stream_task_incremental[2].dataset = concatenated_dataset

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

torch.save(model_task_incre_si.state_dict(), 'model_task_incre_si_label_flipping.pth')

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
      
    #task 4
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task004/Exp003":
      performance_mnist_fellowship.loc["SI","Accuracy - Task 3"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task004/Exp003":
      performance_mnist_fellowship.loc["SI","Loss - Task 3"] = value
    # task 5
    if key == "Top1_Acc_Exp/eval_phase/test_stream/Task005/Exp004":
      performance_mnist_fellowship.loc["SI","Accuracy - Task 4"] = value
      
    if key == "Loss_Exp/eval_phase/test_stream/Task005/Exp004":
      performance_mnist_fellowship.loc["SI","Loss - Task 4"] = value
    # Overall accuracy
    if key == "Top1_Acc_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["SI","Accuracy"] = value
      
    if key == "Loss_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["SI","Loss"] = value
  counter += 1
print(performance_mnist_fellowship)
  

import matplotlib.pyplot as plt
import pandas as pd


# Extract the keys and values from the data dictionary
keys = list(results_mnist_fellowship_si[0].keys())
print(len(keys))
values = [list(d.values()) for d in results_mnist_fellowship_si]
print(values)

# Create a Pandas DataFrame
df = pd.DataFrame(values, columns=keys)


# Extract the keys and values from the data dictionary
keys = list(results_mnist_fellowship_si[0].keys())
values = [list(d.values()) for d in results_mnist_fellowship_si]


"""### Strategy 2: EWC"""

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from torch.optim.adam import Adam
model = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=1000, hidden_layers=2)#, drop_rate=0.15)
#optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9) #adam optimizer
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

torch.save(model.state_dict(), 'model_task_incre_ewc_label_flipping.pth')

"""#### Evaluation"""

print(results_mnist_fellowship_ewc)

##### FINAL Evaluation Code
# Only determine the correct key
counter = 0
for dict_2 in results_mnist_fellowship_ewc:
  
  for k,v in dict_2.items():
    
    
    #task 1
    if k == "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_mnist_fellowship.loc["EWC","Accuracy - Task 0"] = v
      
    if k == "Loss_Exp/eval_phase/test_stream/Task001/Exp000":
      performance_mnist_fellowship.loc["EWC","Loss - Task 0"] = v
    # task 2
    if k == "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_mnist_fellowship.loc["EWC","Accuracy - Task 1"] = v
      
    if k == "Loss_Exp/eval_phase/test_stream/Task002/Exp001":
      performance_mnist_fellowship.loc["EWC","Loss - Task 1"] = value
    # task 3
    if k == "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_mnist_fellowship.loc["EWC","Accuracy - Task 2"] = value
      
    if k == "Loss_Exp/eval_phase/test_stream/Task003/Exp002":
      performance_mnist_fellowship.loc["EWC","Loss - Task 2"] = value
      
    #task 4
    
    # Overall accuracy
    if k == "Top1_Acc_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["EWC","Accuracy"] = value
      
    if k == "Loss_Stream/eval_phase/test_stream/Task003":
      performance_mnist_fellowship.loc["EWC","Loss"] = value
  counter += 1
print(performance_mnist_fellowship)
  
  
torch.save(model.state_dict(), 'model_task_incre_ewc.pth')

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
model_task_incre_online_ewc = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=1000, hidden_layers=2)#, drop_rate=0.15)

optimizer = Adam(model_task_incre_online_ewc.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_online_ewc = OnlineEWC(
    model_task_incre_online_ewc, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""#### Training"""

# TRAINING LOOP
print('Starting experiment...')
results_fellowship_mnist_online_ewc = []
for experience in train_stream_task_incremental:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_online_ewc.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_fellowship_mnist_online_ewc.append(cl_strategy_online_ewc.eval(train_stream_task_incremental))

torch.save(model_task_incre_online_ewc.state_dict(), 'model_task_incre_online_ewc_label_flipping.pth')

"""#### Evaluation"""

print(results_fellowship_mnist_online_ewc)

##### FINAL Evaluation Code
# Only determine the correct key
counter = 0
for dict_1 in results_fellowship_mnist_online_ewc:
  
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
  

performance_mnist_fellowship.to_csv("continual_learning_performance_MNIST_fellowship_label_flipping.csv")

