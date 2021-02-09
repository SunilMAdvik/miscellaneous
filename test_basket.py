
# coding: utf-8

# In[1]:


#get_ipython().magic('reload_ext autoreload')
#get_ipython().magic('autoreload 2')
import glob, random, cv2
import os, sys, argparse, time
sys.path.append('..')

import pickle
import torch
import seaborn as sns
import torch.nn as nn
#Install numpy==1.16
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm_notebook, tqdm
#install skimage==0.15
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import seaborn as sns

from CIFAR.models.wrn import WideResNet 
from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
#import utils.svhn_loader as svhn
import utils.lsun_loader as lsun_loader


# In[2]:


args = {
       'test_bs': 200,
       'num_to_avg': 1, # 'Average measures across num_to_avg runs.' 
       'validate': '', 
       'use_xent': '', 
       'method_name': 'basket_14_wrn_OECC_tune', # 'Method name.basket_14_wrn_OECC_tune_epoch_14.pt
       'layers': 40,
       'widen_factor': 2,
       'droprate': 0.3,
       'load': '/home/ubuntu/efs_model/models/outlier_Detection/results',
       'save': '/home/ubuntu/efs_model/models/outlier_Detection/results',
       'ngpu': 1,
       'prefetch': 4
       }


# In[3]:


root_dir = './'

# train_data_in = []
# test_data_in = []
# data_in = []
# num_classes = 10

# for idx, folder in enumerate(glob.glob('/home/enakshi/Desktop/work/OOD-detection-using-OECC/crop_train/in-distribution/*')):
#     print("data preparing done folder: ", idx, folder.replace('/home/enakshi/Desktop/work/OOD-detection-using-OECC/crop_train/in-distribution/',''))
#     for file in glob.glob(folder + '/*.jpg'):
#         img = cv2.imread(file)
#         img = cv2.resize(img, (32, 32))
#         img = np.transpose(img, (2, 0, 1))
#         img = torch.tensor(img).float()
#         img = (img, idx)
#         data_in.append(img)
# random.shuffle(data_in)
# train_data_in = data_in[0: int(len(data_in) * 0.8)]
# test_data_in = data_in[int(len(data_in) * 0.8):]

# print("in dataset size: ", len(data_in))
# print("in train dataset size: ", len(train_data_in))
# print("in test dataset size: ", len(test_data_in))

# train_loader_in = torch.utils.data.DataLoader(
#     train_data_in, batch_size=args['test_bs'], shuffle=True,
#     num_workers=args['prefetch'], pin_memory=True)
# test_loader = torch.utils.data.DataLoader(
#     test_data_in, batch_size=args['test_bs'], shuffle=False,
#     num_workers=args['prefetch'], pin_memory=True)

#/home/ubuntu/efs_model/models/outlier_Detection/Test_images/outliers
#/home/ubuntu/efs_model/models/outlier_Detection/Test_images/inliers
#/home/ubuntu/efs_model/models/outlier_Detection/Test_images/Inliers_1

start=time.time()
train_data_out = []
test_data_out = []
data_out = []
num_classes = 41

for idx, folder in enumerate(glob.glob('/home/ubuntu/efs_model/models/outlier_Detection/Test_images/Inliers_1')):
    print("data preparing done folder: ", idx, folder.replace('/home/ubuntu/efs_model/models/outlier_Detection/Test_images/Inliers_1',''))
    for file in glob.glob(folder + '/*'):
        print (file)
        img = cv2.imread(file)
        img = cv2.resize(img, (32, 32))
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).float()
        img = (img, idx)
        data_out.append(img)
# random.shuffle(data_out)
train_data_out = data_out[0: int(len(data_out) * 1.0)]
test_data_out = data_out[int(len(data_out) * 1.0):]

print("out dataset size: ", len(data_out))
print("out train dataset size: ", len(train_data_out))
print("out test dataset size: ", len(test_data_out))    
    
train_loader_out = torch.utils.data.DataLoader(
    train_data_out, batch_size=args['test_bs'], shuffle=True,
    num_workers=args['prefetch'], pin_memory=True)

# test_loader_out = torch.utils.data.DataLoader(
#     test_data_out, batch_size=args['test_bs'], shuffle=True,
#     num_workers=args['prefetch'], pin_memory=True)


# In[4]:


# Create model
if 'allconv' in args['method_name']:
    net = AllConvNet(num_classes)
else:
    net = WideResNet(args['layers'],
                     num_classes,
                     args['widen_factor'],
                     dropRate=args['droprate'])
if args['ngpu'] > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args['ngpu'])))
start_epoch = 0
if 'baseline' in args['method_name']:
    subdir = 'baseline'
elif 'OECC' in args['method_name']:
    subdir = 'OECC_tune'
f = open(os.path.join(os.path.join(args['save'], subdir),args['method_name'] + '_test.txt'), 'w+')
# Restore model
if args['load'] != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(
            os.path.join(args['load'], subdir),args['method_name'] + '_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch: ', i)
            f.write('Model restored! Epoch: {}'.format(i))
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"
net.eval()


# In[5]:


cudnn.benchmark = True  # fire on all cylinders
# /////////////// Detection Prelims ///////////////
ood_num_examples = len(train_loader_out) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(train_loader_out))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


# In[6]:


def get_ood_scores(loader, in_dist=False):
    _score = []
    out_conf_score = []
    in_conf_score = []
    _right_score = []
    _wrong_score = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args[
                    'test_bs'] and in_dist is False:
                break
#             data = data.cuda(device)
#             data = data.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            if args['use_xent']:
                _score.append(
                    to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))
                out_conf_score.append(np.max(smax, axis=1))
            if in_dist:
                in_conf_score.append(np.max(smax, axis=1))
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                if args['use_xent']:
                    _right_score.append(
                        to_np((output.mean(1) -
                               torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(
                        to_np((output.mean(1) -
                               torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    if in_dist:
        return concat(in_conf_score).copy(), concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(out_conf_score).copy(), concat(_score)[:ood_num_examples].copy()


# In[7]:


in_conf_score, in_score, right_score, wrong_score = get_ood_scores(train_loader_out, in_dist=True)


# In[8]:


print (in_conf_score,len(in_conf_score))#, in_score, right_score, wrong_score)
for i in in_conf_score:
    print (i)
End = time.time()
print("Time:"End-start)

# In[9]:


"""[0.02440342 0.03000228 0.04045146 0.09346012 0.02439651 0.02439235
 0.02439098 0.02439165 0.02444858 0.02501616 0.0375126  0.02439201
 0.02684388 0.03796991 0.02448195 0.09519789 0.02540645 0.0549598
 0.02439098 0.0244221  0.02641496 0.02439264 0.0253181  0.09519789
 0.02697501 0.02665677 0.02443726 0.06661884 0.06934256 0.02439296
 0.0244042  0.04046609 0.02439201 0.02442326] 34

[0.89110327 0.89010185 0.61668843 0.8897585  0.8706397  0.8910969
 0.892733   0.8732386  0.2229129  0.8317802  0.8035645  0.6376399
 0.3782162  0.8052539  0.85574114 0.87868994 0.09363052 0.8473446
 0.86341155 0.8908998  0.7643845  0.85433096 0.8300025  0.8454769
 0.19021283 0.89159685 0.84640515 0.32551008 0.20913751 0.8666127
 0.14115056 0.7634983  0.5775246  0.8072931  0.5619001  0.79778486
 0.029302   0.19408499 0.6120962  0.8344404  0.03260595 0.6901722
 0.8802659  0.8625703  0.1208948  0.7861998  0.8328867  0.85455114] 48


[0.02439098 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439339 0.02439138 0.02439098 0.02439098 0.02439098 0.02439399
 0.02441355 0.02439098 0.02439098 0.02439098 0.02439453 0.02440926
 0.02439098 0.02439098 0.02439097 0.02439098 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439096 0.02443543 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02457896 0.02439098
 0.02445443 0.02439241 0.02496316 0.02439098 0.02439098 0.02439098
 0.02440254 0.02439098 0.02439098 0.02443244 0.02439098 0.02439335
 0.02439095 0.02439098 0.02439098 0.02439098 0.0243986  0.02439107
 0.02439098 0.02439098 0.02439947 0.02439404 0.02439098 0.02439098
 0.02439098 0.02443295 0.02439238 0.03070659 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439106 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439968 0.02439098 0.02439098 0.02439189
 0.02439095 0.02439098 0.02439126 0.02439098 0.02439098 0.02439098
 0.02439449 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439097 0.0244011  0.02439098 0.02439186
 0.02439098 0.02439098 0.02439098 0.02439095 0.02439098 0.02439098
 0.02439098 0.02439251 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439125 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439096 0.02439944 0.02439098 0.02439095 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439121 0.02439098 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439602 0.02439319 0.02439098 0.02439098
 0.0243909  0.024391   0.02439098 0.02439098 0.02439098 0.02439098
 0.0244039  0.02443613 0.02439128 0.02439098 0.02439098 0.02439098
 0.0244019  0.02474941 0.02439098 0.02439095 0.02439319 0.02439204
 0.02439098 0.02439274 0.02439098 0.02439098 0.02439929 0.02439773
 0.02439098 0.02439098 0.02440083 0.02439098 0.02439097 0.02439547
 0.02444231 0.02439338 0.02439098 0.02439098 0.02439098 0.02447919
 0.02440931 0.02439098 0.02439098 0.02439098 0.02439345 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098 0.02445452
 0.0243958  0.02439098 0.02439098 0.02439177 0.02439098 0.02439098
 0.02439098 0.02439098 0.0243927  0.02439701 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098 0.02439094
 0.02439098 0.02439098 0.02439125 0.02439205 0.02439295 0.02439098
 0.02439098 0.02440684 0.02439098 0.02439098 0.02439098 0.02439098
 0.0246783  0.02439098 0.02439095 0.02439098 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439689 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439299 0.02439098
 0.02439418 0.02439496 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439196 0.02439098 0.02439098
 0.02439515 0.02439098 0.02439098 0.02439098 0.02439097 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439709 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098
 0.0244028  0.02441242 0.02439098 0.02439098 0.02439098 0.02439259
 0.02439098 0.02439098 0.02439359 0.02439174 0.02439098 0.02439098
 0.02439098 0.02439098 0.02439656 0.02439098 0.02439096 0.02439098
 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098 0.02439098] 


# In[9]:


num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
f.write('\nError Rate {:.2f}'.format(100 * num_wrong /
                                     (num_wrong + num_right)))
# /////////////// End Detection Prelims ///////////////
print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print(
    '\nUsing CIFAR-100 as typical data')
f.write('\nUsing CIFAR-10 as typical data') if num_classes == 10 else f.write(
    '\nUsing CIFAR-100 as typical data')


# In[10]:


# /////////////// Error Detection ///////////////

# print('\n\nError Detection')
# f.write('\n\nError Detection')
# show_performance(wrong_score, right_score, f, method_name=args['method_name'])

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []
def get_and_print_results(ood_loader, num_to_avg=args['num_to_avg']):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_conf_score, out_score = get_ood_scores(ood_loader)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0])
        auprs.append(measures[1])
        fprs.append(measures[2])

    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, f, args['method_name'])
    else:
        print_measures(auroc, aupr, fpr, f, args['method_name'])
    return out_conf_score  


# In[11]:


# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args['num_to_avg'])
ood_data = torch.from_numpy(
    np.float32(
        np.clip(
            np.random.normal(size=(ood_num_examples * args['num_to_avg'], 3,
                                   32, 32),
                             scale=0.5), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=args['test_bs'],
                                         shuffle=True,
                                         num_workers=args['prefetch'],
                                         pin_memory=True)

print('\n\nGaussian Noise (sigma = 0.5) Detection')
f.write('\n\nGaussian Noise (sigma = 0.5) Detection')
get_and_print_results(ood_loader)


# In[ ]:


#/////////////// Rademacher Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args['num_to_avg'])
ood_data = torch.from_numpy(
    np.random.binomial(
        n=1, p=0.5, size=(ood_num_examples * args['num_to_avg'], 3, 32,
                          32)).astype(np.float32)) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=args['test_bs'],
                                         shuffle=True)

print('\n\nRademacher Noise Detection')
f.write('\n\nRademacher Noise Detection')
get_and_print_results(ood_loader)


# In[ ]:


# /////////////// Blob ///////////////

ood_data = np.float32(
    np.random.binomial(n=1,
                       p=0.7,
                       size=(ood_num_examples * args['num_to_avg'], 32, 32,
                             3)))
for i in range(ood_num_examples * args['num_to_avg']):
    ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
    ood_data[i][ood_data[i] < 0.75] = 0.0

dummy_targets = torch.ones(ood_num_examples * args['num_to_avg'])
ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=args['test_bs'],
                                         shuffle=True,
                                         num_workers=args['prefetch'],
                                         pin_memory=True)

print('\n\nBlob Detection')
f.write('\n\nBlob Detection')
get_and_print_results(ood_loader)


# In[ ]:


# /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root="./dtd/images",
                            transform=trn.Compose([
                                trn.Resize(32),
                                trn.CenterCrop(32),
                                trn.ToTensor(),
                                trn.Normalize(mean, std)
                            ]))
ood_loader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=args['test_bs'],
                                         shuffle=True,
                                         num_workers=args['prefetch'],
                                         pin_memory=True)

print('\n\nTexture Detection')
f.write('\n\nTexture Detection')
texture_out_score = get_and_print_results(ood_loader)


# In[ ]:


# /////////////// SVHN ///////////////

ood_data = svhn.SVHN(root='SVHN',
                     split="test",
                     transform=trn.Compose([
                         trn.Resize(32),
                         trn.ToTensor(),
                         trn.Normalize(mean, std)
                     ]),
                     download=True)
ood_loader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=args['test_bs'],
                                         shuffle=True,
                                         num_workers=args['prefetch'],
                                         pin_memory=True)

print('\n\nSVHN Detection')
f.write('\n\nSVHN Detection')
svhn_out_score = get_and_print_results(ood_loader)


# In[ ]:


# /////////////// Places365 ///////////////
ood_data = dset.ImageFolder(root="./Places365/",
                            transform=trn.Compose([
                                trn.Resize(32),
                                trn.CenterCrop(32),
                                trn.ToTensor(),
                                trn.Normalize(mean, std)
                            ]))
ood_loader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=args['test_bs'],
                                         shuffle=True,
                                         num_workers=args['prefetch'],
                                         pin_memory=True)

print('\n\nPlaces365 Detection')
f.write('\n\nPlaces365 Detection')
places_out_score = get_and_print_results(ood_loader)


# In[ ]:


# /////////////// LSUN ///////////////

ood_data = lsun_loader.LSUN("./lsun_dataset",
                            classes='test',
                            transform=trn.Compose([
                                trn.Resize(32),
                                trn.CenterCrop(32),
                                trn.ToTensor(),
                                trn.Normalize(mean, std)
                            ]))
ood_loader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=args['test_bs'],
                                         shuffle=True,
                                         num_workers=args['prefetch'],
                                         pin_memory=True)

print('\n\nLSUN Detection')
f.write('\n\nLSUN Detection')
lsun_out_score = get_and_print_results(ood_loader)


# In[ ]:


# /////////////// CIFAR Data ///////////////
train_transform = trn.Compose([
    trn.RandomHorizontalFlip(),
    trn.RandomCrop(32, padding=4),
    trn.ToTensor(),
    trn.Normalize(mean, std)
])
if 'cifar10_' in args['method_name']:
    ood_data = dset.CIFAR100(root_dir,
                             train=False,
                             download=True,
                             transform=train_transform)
else:
    ood_data = dset.CIFAR10(root_dir,
                            train=False,
                            download=True,
                            transform=train_transform)
ood_loader = torch.utils.data.DataLoader(ood_data,
                                         batch_size=args['test_bs'],
                                         shuffle=True,
                                         num_workers=args['prefetch'],
                                         pin_memory=True)
print(
    '\n\nCIFAR-100 Detection') if 'cifar10_' in args['method_name'] else print(
        '\n\nCIFAR-10 Detection')
f.write('\n\nCIFAR-100 Detection'
        ) if 'cifar10_' in args['method_name'] else f.write(
            '\n\nCIFAR-10 Detection')
get_and_print_results(ood_loader)
# /////////////// Mean Results ///////////////
print('\n\nMean Test Results')
f.write('\n\nMean Test Results')
print_measures(np.mean(auroc_list),
               np.mean(aupr_list),
               np.mean(fpr_list),
               f,
               method_name=args['method_name'])
f.close()


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
plt.hist(in_conf_score[:1000], bins=10)
plt.hist(places_out_score[:1000], bins=10)
plt.legend(labels=['cifar10', 'places365'],loc='upper center', prop={'size': 20})
plt.ylabel('Number of examples', fontdict={'fontsize': 20})
plt.xlabel('Softmax probablities', fontdict={'fontsize': 20})
plt.xticks(size=20)
plt.yticks(size=20)
# plt.figure(background_color='white')
plt.savefig('ours_cifar10_places365_distribution.png')"""

