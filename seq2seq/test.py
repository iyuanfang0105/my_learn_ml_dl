# import torch
# import torchvision
# import torchvision.transforms as transforms
#
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.1307,), (0.3081,))])
#
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#
# batch_size_train = 64
# batch_size_test = 1000
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)
#
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
#
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-np.pi,np.pi,256,endpoint=True)
C,S=np.cos(x),np.sin(x)
plt.plot(x,C)
plt.plot(x,S)
plt.show()