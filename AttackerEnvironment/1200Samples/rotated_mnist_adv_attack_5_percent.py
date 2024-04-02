# -*- coding: utf-8 -*-
"""rotated_mnist_adv_attack_5_percent.ipynb
"""

import torch
import torchvision


print("Scenario 1: Domain Incremental - RotatedMNIST")
from avalanche.benchmarks.classic import RotatedMNIST
import pandas as pd

performance_rotated_mnist = pd.DataFrame(index=["SI","EWC","Online EWC"], columns=["Loss - Task 0","Loss - Task 1","Loss - Task 2","Loss - Task 3","Loss - Task 4","Accuracy - Task 0","Accuracy - Task 1","Accuracy - Task 2","Accuracy - Task 3","Accuracy - Task 4","Accuracy", "Loss"])

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from avalanche.benchmarks import dataset_benchmark
import torch.utils.data as data_utils

import random



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



"""# Adversarial Attack - SI

## Adversarial Attack Parameters Initlialization
"""



# Adversarial Attack
import numpy as np
from tqdm import tqdm
import random
## Initialize all the parameters
epsilon = 0.05
T = 240
mu = 1.0
c = 10 # no. of classes
step_size_si = (4*epsilon)/T

"""## Load clean trained mdoel for attack"""

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from avalanche.training.supervised import SynapticIntelligence, EWC

from torch.optim.adam import Adam
model_domain_incre_si = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=400, hidden_layers=2, drop_rate=0.15)

optimizer = Adam(model_domain_incre_si.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

# load the parameters into the model
model_domain_incre_si.load_state_dict(torch.load("AttackerEnvironment/CleanModels/model_domain_incre_si_final.pth"))

cl_strategy_si = SynapticIntelligence(
    model_domain_incre_si, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, si_lambda = 5
)

for param in model_domain_incre_si.parameters():
    param.requires_grad = True

"""## Adversarial Attack Algorithm"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import random

def start_poisoning_trial(model, target_batches, poison_batches, epsilon, iterations, decay_factor, num_classes, step_size):
    original_labels_poison_batch = []
    Xadv_tau_n = [] ### Empty list which WILL CONTAIN list of images perturbed by the algorithm. Images of the poison batch will be perturbed.
    labels_to_be_changed = len(target_batches)*epsilon ### Number of images in target batch * epsilon. This is the number of labels that need to be changed
    labels_to_be_changed_list = random.sample(range(0, len(target_batches)), int(labels_to_be_changed)) ### It is a list of randomly selected labels.
    flipped_labels = [] ### Empty list which WILL CONTAIN flipped labels


    for i, data in enumerate(tqdm(target_batches)):
          poisoned_label = (target_batches[0][1] + torch.randint(1, num_classes-1, (1,))) % num_classes ### Flip a label from the target batch
          flipped_labels.append(poisoned_label) ### Append the flipped label to the apporpriate list
    

    
    momentum = 0
    
    counter = 0
    # Step 4: Compute target gradients Δtθ=∇θL(f(Xτ,θ),Yadvτ)
    for images, labels in target_batches:
        # Convert labels to one-hot encoding
        flipped_labels_individual = flipped_labels[counter] ### Get the flipped label of every corresponding data point
        list_of_labels = torch.eye(10)[flipped_labels_individual].float()
        
        optimizer.zero_grad()  # Reset gradients

        output_probabilities = model(images)

        # Resize the output probabilities to match the batch size of labels
        output_probabilities = output_probabilities.repeat(list_of_labels.size(0), 1)
        
        loss = criterion(output_probabilities, list_of_labels)  # Compute loss

        loss.backward()  # Compute gradients

        target_gradients = [param.grad for param in model.parameters() if param.grad is not None]
        counter = counter + 1
    

    # Step 5-10: Iteratively update the poison samples
    for t in range(iterations):
        print("t = ", t)
        # Step 6: Compute poison gradients Δpθ=∇θL(f(Xadvτ+n,θ),Yτ+n)
        cosine_similarity_accumulator = 0
        counter_poison = 0
        
        for images, labels in poison_batches:
            
            
            X = images.clone().detach().requires_grad_(True)
            # Convert labels to one-hot encoding

            list_of_labels = torch.eye(10)[labels].long()
            
            optimizer.zero_grad()  # Reset gradients

            output_probabilities = model(images)

            # Resize the output probabilities to match the batch size of labels
            output_probabilities = output_probabilities.repeat(list_of_labels.size(0), 1)
            
            #compute loss between 1) output of model for poisoned dataset 2) label for that actual data point without poisoning
            loss = criterion(output_probabilities, list_of_labels)  # Compute loss

            loss.backward()  # Compute gradients

            poison_gradients = [param.grad for param in model.parameters() if param.grad is not None]
            counter_poison = counter_poison + 1
    

        # Step 7: Compute cosine similarity H=⟨Δtθ,Δpθ⟩ / (∥Δtθ∥ ∥Δpθ∥)
            cosine_similarity = torch.dot(torch.cat([g.flatten() for g in target_gradients]),
                                      torch.cat([g.flatten() for g in poison_gradients]))
            norm_target_gradients = torch.norm(torch.cat([g.flatten() for g in target_gradients]))
            norm_poison_gradients = torch.norm(torch.cat([g.flatten() for g in poison_gradients]))
            cosine_similarity /= (norm_target_gradients * norm_poison_gradients)

        # Step 8: Update momentum gt+1=μ⋅gt + ∇X(H/∥H∥)
            
            H = cosine_similarity.clone().requires_grad_(True)
            
            momentum = decay_factor * momentum + (cosine_similarity - cosine_similarity_accumulator / torch.norm(cosine_similarity))

        # Step 9: Update Xadvτ+n = Xadvτ+n + α⋅sign(gt+1)
            images = images + step_size * torch.sign(momentum)
            cosine_similarity_accumulator = cosine_similarity_accumulator + cosine_similarity
            if(t == T - 1):#if if(poison_counter == len(target_batch)-1)
              Xadv_tau_n.append(images)
              original_labels_poison_batch.append(labels)
    # Step 11: Return Xadvτ+n
    return Xadv_tau_n,original_labels_poison_batch

from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.utils.data import ConcatDataset

"""## Perform Adversarial Attack on 40 Degrees Data"""
print("Perform Adversarial Attack on 40 Degrees Data -- SI")
from avalanche.benchmarks.utils.data_attribute import DataAttribute

train_mnist_rotated_40_task1_poisoned,labels_train_mnist_rotated_40 = start_poisoning_trial(model_domain_incre_si, train_mnist_rotated_0, train_mnist_rotated_40, epsilon, T, 1.0, 10, step_size_si)



train_mnist_rotated_40_poisoned_list = []

for i in range(len(train_mnist_rotated_40_task1_poisoned)):
  data = train_mnist_rotated_40_task1_poisoned[i]
  label = labels_train_mnist_rotated_40[i]
  tup = (data,label)
  train_mnist_rotated_40_poisoned_list.append(tup)

"""## Perform Adversarial Attack on 80 Degrees Data"""
print("Perform Adversarial Attack on 80 Degrees Data -- SI")
train_mnist_rotated_80_task1_poisoned,labels_train_mnist_rotated_80 = start_poisoning_trial(model_domain_incre_si, train_mnist_rotated_0, train_mnist_rotated_80, epsilon, T, 1.0, 10, step_size_si)

train_mnist_rotated_80_poisoned_list = []

for i in range(len(train_mnist_rotated_80_task1_poisoned)):
  data = train_mnist_rotated_80_task1_poisoned[i]
  label = labels_train_mnist_rotated_80[i]
  tup = (data,label)
  train_mnist_rotated_80_poisoned_list.append(tup)


"""## Perform Adversarial Attack on 120 Degrees Data"""
print("Perform Adversarial Attack on 120 Degrees Data -- SI")
train_mnist_rotated_120_task1_poisoned,labels_train_mnist_rotated_120 = start_poisoning_trial(model_domain_incre_si, train_mnist_rotated_0, train_mnist_rotated_120, epsilon, T, 1.0, 10, step_size_si)

train_mnist_rotated_120_poisoned_list = []

for i in range(len(train_mnist_rotated_120_task1_poisoned)):
  data = train_mnist_rotated_120_task1_poisoned[i]
  label = labels_train_mnist_rotated_120[i]
  tup = (data,label)
  train_mnist_rotated_120_poisoned_list.append(tup)


"""## Perform Adversarial Attack on 160 Degrees Data"""
print("Perform Adversarial Attack on 160 Degrees Data -- SI")
train_mnist_rotated_160_task1_poisoned,labels_train_mnist_rotated_160 = start_poisoning_trial(model_domain_incre_si, train_mnist_rotated_0, train_mnist_rotated_160, epsilon, T, 1.0, 10, step_size_si)

train_mnist_rotated_160_poisoned_list = []

for i in range(len(train_mnist_rotated_160_task1_poisoned)):
  data = train_mnist_rotated_160_task1_poisoned[i]
  label = labels_train_mnist_rotated_160[i]
  tup = (data,label)
  train_mnist_rotated_160_poisoned_list.append(tup)


"""## Processing data for Creation of scenario"""

task_labels_train = [1] * 1200

task_labels_test = [1] * 100

"""DataAttribute:
a class designed to managed task and class labels. DataAttributes allow fast
concatenation and subsampling operations and are automatically managed by
AvalancheDatasets
"""

from avalanche.benchmarks.utils.data_attribute import DataAttribute
#extract labels from datasets (we will put those in ClassificationDataset.targets)

poisoned_train_mnist_rotated_40_labels_final = DataAttribute(labels_train_mnist_rotated_40, "test40")

poisoned_train_mnist_rotated_80_labels_final = DataAttribute(labels_train_mnist_rotated_80, "test80")

poisoned_train_mnist_rotated_120_labels_final = DataAttribute(labels_train_mnist_rotated_120, "test120")

poisoned_train_mnist_rotated_160_labels_final = DataAttribute(labels_train_mnist_rotated_160, "test160")

poisoned_train_mnist_rotated_40 = ClassificationDataset([train_mnist_rotated_40_poisoned_list])
poisoned_train_mnist_rotated_40.targets = poisoned_train_mnist_rotated_40_labels_final
poisoned_train_mnist_rotated_40.targets_task_labels = task_labels_train
test_mnist_rotated_40.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_80 = ClassificationDataset([train_mnist_rotated_80_poisoned_list])
poisoned_train_mnist_rotated_80.targets = poisoned_train_mnist_rotated_80_labels_final
poisoned_train_mnist_rotated_80.targets_task_labels = task_labels_train
test_mnist_rotated_80.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_120 = ClassificationDataset([train_mnist_rotated_120_poisoned_list])
poisoned_train_mnist_rotated_120.targets = poisoned_train_mnist_rotated_120_labels_final
poisoned_train_mnist_rotated_120.targets_task_labels = task_labels_train
test_mnist_rotated_120.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_160 = ClassificationDataset([train_mnist_rotated_160_poisoned_list])
poisoned_train_mnist_rotated_160.targets = poisoned_train_mnist_rotated_160_labels_final
poisoned_train_mnist_rotated_160.targets_task_labels = task_labels_train
test_mnist_rotated_160.targets_task_labels = task_labels_test

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


experience0 = train_stream[0]
dataset_0 = experience0.dataset
#print(len(list(dataset_0)))


experience1 = train_stream[1]
dataset_1 = experience1.dataset


experience2 = train_stream[2]
dataset_2 = experience2.dataset


experience3 = train_stream[3]
dataset_3 = experience3.dataset


experience4 = train_stream[4]
dataset_4 = experience4.dataset

## Here we have to take samples from task 0, change their label and inject them into other tasks


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

for param in model_domain_incre_si.parameters():
    param.requires_grad = True

"""### Training"""

# TRAINING LOOP
print('Starting experiment...')
results_rotated_mnist_si = []
counter = 0
for experience in train_stream:
    #tamper experience here
    #train_stream[counter] =
    #experience0.dataset_0 = concatenated_dataset
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy_si.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results_rotated_mnist_si.append(cl_strategy_si.eval(test_stream))

"""### Evaluation"""

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
  

"""### Saving the Poisoned Model"""

torch.save(model_domain_incre_si.state_dict(), 'model_domain_incre_si_adv_attack_5_percent.pth')

print("############################### SI Completed ##############################")

"""# Adversarial Attack - EWC

## Adversarial Attack Parameters Initlialization
"""


# Adversarial Attack
import numpy as np
from tqdm import tqdm
import random
## Initialize all the parameters
epsilon = 0.05
T = 240
mu = 1.0
c = 10 # no. of classes
step_size_ewc = 2/255

"""## Load clean trained mdoel for attack"""

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from avalanche.training.supervised import SynapticIntelligence, EWC

from torch.optim.adam import Adam
model_domain_incre_ewc = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=400, hidden_layers=2, drop_rate=0.15)
#optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9) #adam optimizer
optimizer = Adam(model_domain_incre_ewc.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

# load the parameters into the model
model_domain_incre_ewc.load_state_dict(torch.load("AttackerEnvironment/CleanModels/model_domain_incre_ewc_final.pth"))

cl_strategy_ewc = EWC(
    model_domain_incre_ewc, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""## Adversarial Attack Algorithm"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import random

def start_poisoning_trial(model, target_batches, poison_batches, epsilon, iterations, decay_factor, num_classes, step_size):
    original_labels_poison_batch = []
    Xadv_tau_n = [] ### Empty list which WILL CONTAIN list of images perturbed by the algorithm. Images of the poison batch will be perturbed.
    labels_to_be_changed = len(target_batches)*epsilon ### Number of images in target batch * epsilon. This is the number of labels that need to be changed
    labels_to_be_changed_list = random.sample(range(0, len(target_batches)), int(labels_to_be_changed)) ### It is a list of randomly selected labels.
    flipped_labels = [] ### Empty list which WILL CONTAIN flipped labels


    for i, data in enumerate(tqdm(target_batches)):
          poisoned_label = (target_batches[0][1] + torch.randint(1, num_classes-1, (1,))) % num_classes ### Flip a label from the target batch
          flipped_labels.append(poisoned_label) ### Append the flipped label to the apporpriate list
    

    
    momentum = 0
    
    counter = 0
    # Step 4: Compute target gradients Δtθ=∇θL(f(Xτ,θ),Yadvτ)
    for images, labels in target_batches:
        # Convert labels to one-hot encoding
        flipped_labels_individual = flipped_labels[counter] ### Get the flipped label of every corresponding data point
        list_of_labels = torch.eye(10)[flipped_labels_individual].float()
        
        optimizer.zero_grad()  # Reset gradients

        output_probabilities = model(images)

        # Resize the output probabilities to match the batch size of labels
        output_probabilities = output_probabilities.repeat(list_of_labels.size(0), 1)
        
        loss = criterion(output_probabilities, list_of_labels)  # Compute loss

        loss.backward()  # Compute gradients

        target_gradients = [param.grad for param in model.parameters() if param.grad is not None]
        counter = counter + 1
    

    # Step 5-10: Iteratively update the poison samples
    for t in range(iterations):
        print("t = ", t)
        # Step 6: Compute poison gradients Δpθ=∇θL(f(Xadvτ+n,θ),Yτ+n)
        cosine_similarity_accumulator = 0
        counter_poison = 0
        
        for images, labels in poison_batches:
            
            
            X = images.clone().detach().requires_grad_(True)
            # Convert labels to one-hot encoding

            list_of_labels = torch.eye(10)[labels].long()
            
            optimizer.zero_grad()  # Reset gradients

            output_probabilities = model(images)

            # Resize the output probabilities to match the batch size of labels
            output_probabilities = output_probabilities.repeat(list_of_labels.size(0), 1)
            
            #compute loss between 1) output of model for poisoned dataset 2) label for that actual data point without poisoning
            loss = criterion(output_probabilities, list_of_labels)  # Compute loss

            loss.backward()  # Compute gradients

            poison_gradients = [param.grad for param in model.parameters() if param.grad is not None]
            counter_poison = counter_poison + 1
    

        # Step 7: Compute cosine similarity H=⟨Δtθ,Δpθ⟩ / (∥Δtθ∥ ∥Δpθ∥)
            cosine_similarity = torch.dot(torch.cat([g.flatten() for g in target_gradients]),
                                      torch.cat([g.flatten() for g in poison_gradients]))
            norm_target_gradients = torch.norm(torch.cat([g.flatten() for g in target_gradients]))
            norm_poison_gradients = torch.norm(torch.cat([g.flatten() for g in poison_gradients]))
            cosine_similarity /= (norm_target_gradients * norm_poison_gradients)

        # Step 8: Update momentum gt+1=μ⋅gt + ∇X(H/∥H∥)
            
            H = cosine_similarity.clone().requires_grad_(True)
            
            momentum = decay_factor * momentum + (cosine_similarity - cosine_similarity_accumulator / torch.norm(cosine_similarity))

        # Step 9: Update Xadvτ+n = Xadvτ+n + α⋅sign(gt+1)
            images = images + step_size * torch.sign(momentum)
            cosine_similarity_accumulator = cosine_similarity_accumulator + cosine_similarity
            if(t == T - 1):#if if(poison_counter == len(target_batch)-1)
              Xadv_tau_n.append(images)
              original_labels_poison_batch.append(labels)
    # Step 11: Return Xadvτ+n
    return Xadv_tau_n,original_labels_poison_batch

from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.utils.data import ConcatDataset

"""## Perform Adversarial Attack on 40 Degrees Data"""
print("Perform Adversarial Attack on 40 Degrees Data -- EWC")
from avalanche.benchmarks.utils.data_attribute import DataAttribute

train_mnist_rotated_40_task1_poisoned,labels_train_mnist_rotated_40 = start_poisoning_trial(model_domain_incre_ewc, train_mnist_rotated_0, train_mnist_rotated_40, epsilon, T, 1.0, 10, step_size_ewc)



train_mnist_rotated_40_poisoned_list = []

for i in range(len(train_mnist_rotated_40_task1_poisoned)):
  data = train_mnist_rotated_40_task1_poisoned[i]
  label = labels_train_mnist_rotated_40[i]
  tup = (data,label)
  train_mnist_rotated_40_poisoned_list.append(tup)


"""## Perform Adversarial Attack on 80 Degrees Data"""
print("Perform Adversarial Attack on 80 Degrees Data -- EWC")
train_mnist_rotated_80_task1_poisoned,labels_train_mnist_rotated_80 = start_poisoning_trial(model_domain_incre_ewc, train_mnist_rotated_0, train_mnist_rotated_80, epsilon, T, 1.0, 10, step_size_ewc)

train_mnist_rotated_80_poisoned_list = []

for i in range(len(train_mnist_rotated_80_task1_poisoned)):
  data = train_mnist_rotated_80_task1_poisoned[i]
  label = labels_train_mnist_rotated_80[i]
  tup = (data,label)
  train_mnist_rotated_80_poisoned_list.append(tup)


"""## Perform Adversarial Attack on 120 Degrees Data"""
print("Perform Adversarial Attack on 120 Degrees Data -- EWC")
train_mnist_rotated_120_task1_poisoned,labels_train_mnist_rotated_120 = start_poisoning_trial(model_domain_incre_ewc, train_mnist_rotated_0, train_mnist_rotated_120, epsilon, T, 1.0, 10, step_size_ewc)

train_mnist_rotated_120_poisoned_list = []

for i in range(len(train_mnist_rotated_120_task1_poisoned)):
  data = train_mnist_rotated_120_task1_poisoned[i]
  label = labels_train_mnist_rotated_120[i]
  tup = (data,label)
  train_mnist_rotated_120_poisoned_list.append(tup)


"""## Perform Adversarial Attack on 160 Degrees Data"""
print("Perform Adversarial Attack on 160 Degrees Data -- EWC")
train_mnist_rotated_160_task1_poisoned,labels_train_mnist_rotated_160 = start_poisoning_trial(model_domain_incre_ewc, train_mnist_rotated_0, train_mnist_rotated_160, epsilon, T, 1.0, 10, step_size_ewc)

train_mnist_rotated_160_poisoned_list = []

for i in range(len(train_mnist_rotated_160_task1_poisoned)):
  data = train_mnist_rotated_160_task1_poisoned[i]
  label = labels_train_mnist_rotated_160[i]
  tup = (data,label)
  train_mnist_rotated_160_poisoned_list.append(tup)




"""## Processing data for Creation of scenario"""

task_labels_train = [1] * 1200

task_labels_test = [1] * 200

from avalanche.benchmarks.utils.data_attribute import DataAttribute
#extract labels from datasets (we will put those in ClassificationDataset.targets)

poisoned_train_mnist_rotated_40_labels_final = DataAttribute(labels_train_mnist_rotated_40,"test40")


poisoned_train_mnist_rotated_80_labels_final = DataAttribute(labels_train_mnist_rotated_80,"test80")


poisoned_train_mnist_rotated_120_labels_final = DataAttribute(labels_train_mnist_rotated_120,"test120")


poisoned_train_mnist_rotated_160_labels_final = DataAttribute(labels_train_mnist_rotated_160,"test160")

poisoned_train_mnist_rotated_40 = ClassificationDataset([train_mnist_rotated_40_poisoned_list])
poisoned_train_mnist_rotated_40.targets = poisoned_train_mnist_rotated_40_labels_final
poisoned_train_mnist_rotated_40.targets_task_labels = task_labels_train
test_mnist_rotated_40.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_80 = ClassificationDataset([train_mnist_rotated_80_poisoned_list])
poisoned_train_mnist_rotated_80.targets = poisoned_train_mnist_rotated_80_labels_final
poisoned_train_mnist_rotated_80.targets_task_labels = task_labels_train
test_mnist_rotated_80.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_120 = ClassificationDataset([train_mnist_rotated_120_poisoned_list])
poisoned_train_mnist_rotated_120.targets = poisoned_train_mnist_rotated_120_labels_final
poisoned_train_mnist_rotated_120.targets_task_labels = task_labels_train
test_mnist_rotated_120.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_160 = ClassificationDataset([train_mnist_rotated_160_poisoned_list])
poisoned_train_mnist_rotated_160.targets = poisoned_train_mnist_rotated_160_labels_final
poisoned_train_mnist_rotated_160.targets_task_labels = task_labels_train
test_mnist_rotated_160.targets_task_labels = task_labels_test

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
    #task_info.task_label = counter
    #print("Task Label:", {task_label})
    counter += counter



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


experience0 = train_stream[0]
dataset_0 = experience0.dataset



experience1 = train_stream[1]
dataset_1 = experience1.dataset


experience2 = train_stream[2]
dataset_2 = experience2.dataset


experience3 = train_stream[3]
dataset_3 = experience3.dataset


experience4 = train_stream[4]
dataset_4 = experience4.dataset

## Here we have to take samples from task 0, change their label and inject them into other tasks


from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import SynapticIntelligence, EWC  # and many more!

from avalanche.training.supervised import SynapticIntelligence, EWC

from torch.optim.adam import Adam
model_domain_incre_ewc = SimpleMLP(num_classes=10, input_size=int(100352/128),  hidden_size=400, hidden_layers=2, drop_rate=0.15)

optimizer = Adam(model_domain_incre_ewc.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_ewc = EWC(
    model_domain_incre_ewc, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""### Training"""

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

"""### Evaluation"""

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
  

"""### Saving the poisoned model"""

torch.save(model_domain_incre_ewc.state_dict(), 'model_domain_incre_ewc_adv_attack_5_percent.pth')

print("############################ EWC ENDS ##########################")

"""# Adversarial Attack - Online EWC

## Adversarial Attack Parameters Initlialization
"""

step_size_online_ewc = (4*epsilon)/T

# Adversarial Attack
import numpy as np
from tqdm import tqdm
import random
## Initialize all the parameters
epsilon = 0.05
T = 240
mu = 1.0
c = 10 # no. of classes
step_size_online_ewc = (4*epsilon)/T

"""## Load clean trained mdoel for attack"""



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

# load the parameters into the model
model_domain_incre_online_ewc.load_state_dict(torch.load("AttackerEnvironment/CleanModels/model_domain_incre_online_ewc_final.pth"))

cl_strategy_online_ewc = OnlineEWC(
    model_domain_incre_online_ewc, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""## Adversarial Attack Algorithm"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import random

def start_poisoning_trial(model, target_batches, poison_batches, epsilon, iterations, decay_factor, num_classes, step_size):
    original_labels_poison_batch = []
    Xadv_tau_n = [] ### Empty list which WILL CONTAIN list of images perturbed by the algorithm. Images of the poison batch will be perturbed.
    labels_to_be_changed = len(target_batches)*epsilon ### Number of images in target batch * epsilon. This is the number of labels that need to be changed
    labels_to_be_changed_list = random.sample(range(0, len(target_batches)), int(labels_to_be_changed)) ### It is a list of randomly selected labels.
    flipped_labels = [] ### Empty list which WILL CONTAIN flipped labels


    for i, data in enumerate(tqdm(target_batches)):
          poisoned_label = (target_batches[0][1] + torch.randint(1, num_classes-1, (1,))) % num_classes ### Flip a label from the target batch
          flipped_labels.append(poisoned_label) ### Append the flipped label to the apporpriate list
    

    
    momentum = 0
    
    counter = 0
    # Step 4: Compute target gradients Δtθ=∇θL(f(Xτ,θ),Yadvτ)
    for images, labels in target_batches:
        # Convert labels to one-hot encoding
        flipped_labels_individual = flipped_labels[counter] ### Get the flipped label of every corresponding data point
        list_of_labels = torch.eye(10)[flipped_labels_individual].float()
        
        optimizer.zero_grad()  # Reset gradients

        output_probabilities = model(images)

        # Resize the output probabilities to match the batch size of labels
        output_probabilities = output_probabilities.repeat(list_of_labels.size(0), 1)
        
        loss = criterion(output_probabilities, list_of_labels)  # Compute loss

        loss.backward()  # Compute gradients

        target_gradients = [param.grad for param in model.parameters() if param.grad is not None]
        counter = counter + 1
    

    # Step 5-10: Iteratively update the poison samples
    for t in range(iterations):
        print("t = ", t)
        # Step 6: Compute poison gradients Δpθ=∇θL(f(Xadvτ+n,θ),Yτ+n)
        cosine_similarity_accumulator = 0
        counter_poison = 0
        
        for images, labels in poison_batches:
            
            
            X = images.clone().detach().requires_grad_(True)
            # Convert labels to one-hot encoding

            list_of_labels = torch.eye(10)[labels].long()
            
            optimizer.zero_grad()  # Reset gradients

            output_probabilities = model(images)

            # Resize the output probabilities to match the batch size of labels
            output_probabilities = output_probabilities.repeat(list_of_labels.size(0), 1)
            
            #compute loss between 1) output of model for poisoned dataset 2) label for that actual data point without poisoning
            loss = criterion(output_probabilities, list_of_labels)  # Compute loss

            loss.backward()  # Compute gradients

            poison_gradients = [param.grad for param in model.parameters() if param.grad is not None]
            counter_poison = counter_poison + 1
    

        # Step 7: Compute cosine similarity H=⟨Δtθ,Δpθ⟩ / (∥Δtθ∥ ∥Δpθ∥)
            cosine_similarity = torch.dot(torch.cat([g.flatten() for g in target_gradients]),
                                      torch.cat([g.flatten() for g in poison_gradients]))
            norm_target_gradients = torch.norm(torch.cat([g.flatten() for g in target_gradients]))
            norm_poison_gradients = torch.norm(torch.cat([g.flatten() for g in poison_gradients]))
            cosine_similarity /= (norm_target_gradients * norm_poison_gradients)

        # Step 8: Update momentum gt+1=μ⋅gt + ∇X(H/∥H∥)
            
            H = cosine_similarity.clone().requires_grad_(True)
            
            momentum = decay_factor * momentum + (cosine_similarity - cosine_similarity_accumulator / torch.norm(cosine_similarity))

        # Step 9: Update Xadvτ+n = Xadvτ+n + α⋅sign(gt+1)
            images = images + step_size * torch.sign(momentum)
            cosine_similarity_accumulator = cosine_similarity_accumulator + cosine_similarity
            if(t == T - 1):#if if(poison_counter == len(target_batch)-1)
              Xadv_tau_n.append(images)
              original_labels_poison_batch.append(labels)
    # Step 11: Return Xadvτ+n
    return Xadv_tau_n,original_labels_poison_batch

from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.utils.data import ConcatDataset

"""## Perform Adversarial Attack on 40 Degrees Data"""
print("Perform Adversarial Attack on 40 Degrees Data -- Online EWC")
from avalanche.benchmarks.utils.data_attribute import DataAttribute

train_mnist_rotated_40_task1_poisoned,labels_train_mnist_rotated_40 = start_poisoning_trial(model_domain_incre_online_ewc, train_mnist_rotated_0, train_mnist_rotated_40, epsilon, T, 1.0, 10, step_size_online_ewc)


train_mnist_rotated_40_poisoned_list = []

for i in range(len(train_mnist_rotated_40_task1_poisoned)):
  data = train_mnist_rotated_40_task1_poisoned[i]
  label = labels_train_mnist_rotated_40[i]
  tup = (data,label)
  train_mnist_rotated_40_poisoned_list.append(tup)


"""## Perform Adversarial Attack on 80 Degrees Data"""
print("Perform Adversarial Attack on 80 Degrees Data -- Online EWC")
train_mnist_rotated_80_task1_poisoned,labels_train_mnist_rotated_80 = start_poisoning_trial(model_domain_incre_online_ewc, train_mnist_rotated_0, train_mnist_rotated_80, epsilon, T, 1.0, 10, step_size_online_ewc)

train_mnist_rotated_80_poisoned_list = []

for i in range(len(train_mnist_rotated_80_task1_poisoned)):
  data = train_mnist_rotated_80_task1_poisoned[i]
  label = labels_train_mnist_rotated_80[i]
  tup = (data,label)
  train_mnist_rotated_80_poisoned_list.append(tup)


"""## Perform Adversarial Attack on 120 Degrees Data"""
print("Perform Adversarial Attack on 120 Degrees Data -- Online EWC")
train_mnist_rotated_120_task1_poisoned,labels_train_mnist_rotated_120 = start_poisoning_trial(model_domain_incre_online_ewc, train_mnist_rotated_0, train_mnist_rotated_120, epsilon, T, 1.0, 10, step_size_online_ewc)

train_mnist_rotated_120_poisoned_list = []

for i in range(len(train_mnist_rotated_120_task1_poisoned)):
  data = train_mnist_rotated_120_task1_poisoned[i]
  label = labels_train_mnist_rotated_120[i]
  tup = (data,label)
  train_mnist_rotated_120_poisoned_list.append(tup)
print(len(train_mnist_rotated_120_poisoned_list))



"""## Perform Adversarial Attack on 160 Degrees Data"""
print("Perform Adversarial Attack on 160 Degrees Data -- Online EWC")
train_mnist_rotated_160_task1_poisoned,labels_train_mnist_rotated_160 = start_poisoning_trial(model_domain_incre_online_ewc, train_mnist_rotated_0, train_mnist_rotated_160, epsilon, T, 1.0, 10, step_size_online_ewc)

train_mnist_rotated_160_poisoned_list = []

for i in range(len(train_mnist_rotated_160_task1_poisoned)):
  data = train_mnist_rotated_160_task1_poisoned[i]
  label = labels_train_mnist_rotated_160[i]
  tup = (data,label)
  train_mnist_rotated_160_poisoned_list.append(tup)


"""## Processing data for Creation of scenario"""

task_labels_train = [1] * 1200

task_labels_test = [1] * 200

from avalanche.benchmarks.utils.data_attribute import DataAttribute
#extract labels from datasets (we will put those in ClassificationDataset.targets)

poisoned_train_mnist_rotated_40_labels_final = DataAttribute(labels_train_mnist_rotated_40,"test40")

poisoned_train_mnist_rotated_80_labels_final = DataAttribute(labels_train_mnist_rotated_80,"test80")

poisoned_train_mnist_rotated_120_labels_final = DataAttribute(labels_train_mnist_rotated_120,"test120")

poisoned_train_mnist_rotated_160_labels_final = DataAttribute(labels_train_mnist_rotated_160,"test160")

poisoned_train_mnist_rotated_40 = ClassificationDataset([train_mnist_rotated_40_poisoned_list])
poisoned_train_mnist_rotated_40.targets = poisoned_train_mnist_rotated_40_labels_final
poisoned_train_mnist_rotated_40.targets_task_labels = task_labels_train
test_mnist_rotated_40.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_80 = ClassificationDataset([train_mnist_rotated_80_poisoned_list])
poisoned_train_mnist_rotated_80.targets = poisoned_train_mnist_rotated_80_labels_final
poisoned_train_mnist_rotated_80.targets_task_labels = task_labels_train
test_mnist_rotated_80.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_120 = ClassificationDataset([train_mnist_rotated_120_poisoned_list])
poisoned_train_mnist_rotated_120.targets = poisoned_train_mnist_rotated_120_labels_final
poisoned_train_mnist_rotated_120.targets_task_labels = task_labels_train
test_mnist_rotated_120.targets_task_labels = task_labels_test


poisoned_train_mnist_rotated_160 = ClassificationDataset([train_mnist_rotated_160_poisoned_list])
poisoned_train_mnist_rotated_160.targets = poisoned_train_mnist_rotated_160_labels_final
poisoned_train_mnist_rotated_160.targets_task_labels = task_labels_train
test_mnist_rotated_160.targets_task_labels = task_labels_test

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
    #task_info.task_label = counter
    #print("Task Label:", {task_label})
    counter += counter



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


experience0 = train_stream[0]
dataset_0 = experience0.dataset

experience1 = train_stream[1]
dataset_1 = experience1.dataset


experience2 = train_stream[2]
dataset_2 = experience2.dataset


experience3 = train_stream[3]
dataset_3 = experience3.dataset


experience4 = train_stream[4]
dataset_4 = experience4.dataset

## Here we have to take samples from task 0, change their label and inject them into other tasks


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
#optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9) #adam optimizer
optimizer = Adam(model_domain_incre_online_ewc.parameters(), lr=0.0001)
criterion = CrossEntropyLoss()

cl_strategy_online_ewc = OnlineEWC(
    model_domain_incre_online_ewc, optimizer, criterion,
    train_mb_size=128, train_epochs=10, eval_mb_size=128, ewc_lambda = 5000
)

"""### Training"""

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

"""### Evaluation"""

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


"""### Saving the Poisoned Model"""

torch.save(model_domain_incre_online_ewc.state_dict(), 'model_domain_incre_online_ewc_adv_attack_5_percent.pth')

"""# Saving the Performance Metrics"""

performance_rotated_mnist.to_csv("continual_learning_performance_RotatedMNIST_5_percent_1200_samples.csv")
