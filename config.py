USE_GPU = True

import torch
dtype = torch.float64 # we will be using float throughout this tutorial

device = torch.device('cuda') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')
print('using device:', device)