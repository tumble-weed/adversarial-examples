tensor_to_numpy = lambda t:t.detach().cpu().numpy()
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from argparse import Namespace
from network import Net
# main()
# Accuracy counter


epsilons = [0, .05, .1, .15, .2, .25, .3, .5]
correct = 0
adv_examples = []

# Loop over all examples in test set
# for data, target in test_loader:
if True or 'clean_data' not in locals():
    test_iter = iter(test_loader)
    clean_data,target = next(test_iter)
data,target = clean_data[:1],target[:1]
print(target)
# assert False
# Send the data and label to the device
data, target = data.to(device), target.to(device)
clean_data = data
n_examples = 1000
data = torch.cat([data.detach().clone() + torch.randn(data.shape,device=device) for _ in range(n_examples)],dim=0)
target = torch.cat([target.detach().clone() for _ in range(n_examples)],dim=0)
# Set requires_grad attribute of tensor. Important for Attack
data.requires_grad = True

# Forward pass the data through the model
output = model(data)
init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

# Calculate the loss
loss = F.nll_loss(output, target)
# loss = - F.nll_loss(output, 8*torch.ones_like(target))
# Zero all existing gradients
model.zero_grad()

# Calculate gradients of model in backward pass
loss.backward()

# Collect datagrad
data_grad = data.grad.data

# Call FGSM Attack
perturbed_data = fgsm_attack(data, epsilons[-1], data_grad)

# Re-classify the perturbed image
output = model(perturbed_data)

# Check for success
final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
print(target,final_pred)
'''
if final_pred.item() == target.item():
    correct += 1
    # Special case for saving 0 epsilon examples
    if (epsilons[-1] == 0) and (len(adv_examples) < 5):
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
else:
    # Save some adv examples for visualization later
    if len(adv_examples) < 5:
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
'''

if 'visualize':
    diff = (perturbed_data - data)[final_pred.squeeze()!=target]
    diff_ = tensor_to_numpy(diff)
    data_= tensor_to_numpy(data)
    clean_data_ = tensor_to_numpy(clean_data)
    plt.figure()
    plt.imshow(clean_data_[0,0])
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(data_[0,0])
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(diff_[0,0])
    plt.show()
    plt.close()
    mean_diff_ = diff_.mean(axis = 0,keepdims=True)

    plt.figure()
    plt.imshow(mean_diff_[0,0])
    plt.show()
    plt.close()

    plt.figure()
    plt.imshow(1.35*mean_diff_[0,0] + clean_data_[0,0])
    plt.show()
    plt.close()
