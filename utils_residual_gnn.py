from decimal import DivisionByZero
import random
# from ranger import Ranger
from networkx import center
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import dgl
from dgl.data.utils import Subset

from center_loss import CenterLoss
import settings as st

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#   print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

# @title Set device (GPU or CPU). Execute `set_device()`
# especially if torch modules used.
# Inform the user if the notebook uses GPU or CPU.
def set_device():
  """
  Set the device. CUDA if available, CPU otherwise

  Args:
    None

  Returns:
    Nothing
  """
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
#   if device != "cuda:0":
#     print("WARNING: For this notebook to perform best, "
#         "if possible, in the menu under `Runtime` -> "
#         "`Change runtime type.`  select `GPU` ")
#   else:
#     print("GPU is enabled in this py code.")

  return device

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def myoptimizer(net, opt_parameters):

    # Backward pass
    lr = opt_parameters['lr']
    wd = opt_parameters['weight_decay']
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # optimizer = Ranger(params=net.parameters(), lr=lr)

    return optimizer

def convert_lst_to_batch(x_batch):

    landmark_lst = []
    num_landmarks = len(x_batch[0])

    for i in range(num_landmarks):
       landmark = [item[i] for item in x_batch]
       landmark_lst.append(torch.stack(landmark, dim=0))

    return landmark_lst

def features_collate4metric(samples):

    x1_batch, x2_batch, labels, parent_ids, child_ids = map(list, zip(*samples))
    labels = torch.LongTensor(labels)
    parent_ids = torch.LongTensor(parent_ids)
    child_ids = torch.LongTensor(child_ids)

    x1_batch = torch.Tensor(np.array(x1_batch))
    x2_batch = torch.Tensor(np.array(x2_batch))

    return x1_batch, x2_batch, labels, parent_ids, child_ids

def features_collate(samples):

    x1_batch, x2_batch, labels = map(list, zip(*samples))
    labels = torch.tensor(labels)

    x1_batch = torch.Tensor(np.array(x1_batch))
    x2_batch = torch.Tensor(np.array(x2_batch))

    return x1_batch, x2_batch, labels

def collate(samples):

    x1_batch, x2_batch, labels = map(list, zip(*samples))
    labels = torch.tensor(labels)

    x1_batch = torch.stack(convert_lst_to_batch(x1_batch))
    x2_batch = torch.stack(convert_lst_to_batch(x2_batch))

    return x1_batch, x2_batch, labels

def first_collate(samples):

    x1_batch, x2_batch, labels = map(list, zip(*samples))
    labels = torch.tensor(labels)

    x1_batch = convert_lst_to_batch(x1_batch)
    x2_batch = convert_lst_to_batch(x2_batch)

    return x1_batch, x2_batch, labels

# Compute accuracy
def accuracy(logits, targets):

    # pred = (logits > 0.5).long()
    # results = (pred == (targets.long())).long()
    # results = results.cpu().data.numpy()
    # acc = sum(results) * 1.0 / len(results)
    # return acc

    preds = logits.detach().argmax(dim=1)
    acc = (preds==targets).sum().item()
    return acc

def confusion_matrix(logits, targets):
    pred_labels = logits.detach().argmax(dim=1)

    # general strategy
    # use the “AND” operator to combine the results into a single binary vector
    # sum over the binary vector to count how many incidences there are

    # True Positive (TP): we predict a label of 1 (positive), and the actual label is 1.
    # TP = np.sum(np.logical_and(pred_labels == 1, targets == 1))
    TP = torch.logical_and(pred_labels == 1, targets == 1).sum()

    # True Negative (TN): we predict a label of 0 (negative), and the actual label is 0.
    # TN = np.sum(np.logical_and(pred_labels == 0, targets == 0))
    TN = torch.logical_and(pred_labels == 0, targets == 0).sum()

    # False Positive (FP): we predict a label of 1 (positive), but the actual label is 0.py
    # FP = np.sum(np.logical_and(pred_labels == 1, targets == 0))
    FP = torch.logical_and(pred_labels == 1, targets == 0).sum()

    # False Negative (FN): we predict a label of 0 (negative), but the actual label is 1.
    # FN = np.sum(np.logical_and(pred_labels == 0, targets == 1))
    FN = torch.logical_and(pred_labels == 0, targets == 1).sum()

    return TN, FP,FN, TP

def save_checkpoint(model, optimizer, save_path, epoch, kin_type, batch_size, fold_num):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'kin_type':kin_type,
        'batch_size':batch_size,
        'fold_num':fold_num
    }, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()

        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # dist_ap.append(dist[i][mask[i]].max())
            # dist_an.append(dist[i][mask[i] == 0].min())
            dist_ap.append(torch.tensor([float(dist[i][mask[i]].max())]))
            dist_an.append(torch.tensor([float(dist[i][mask[i] == 0].min())]))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss


class Criterion:
    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        # self.triplest = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.triplest = TripletLoss()
        self.cross_loss = nn.CrossEntropyLoss()
        # self.xenter_loss = CenterContrastiveLoss(num_classes=2, feat_dim=st.FC1)
        self.xenter_loss = CenterLoss(num_classes=2, feat_dim=st.config[st.I]["FC1"])
        self.optimizer_xentloss = torch.optim.Adam(self.xenter_loss.parameters(), lr=0.5)
        return 

    def criterion1(self, batch_scores, gt_all):
        return self.bce(batch_scores, gt_all)  # pos_weight=torch.tensor(1)

    def criterion2(self, f_parent, f_child):
        return self.mse(f_parent, f_child)

    def criterion3(self, parent_child_features, target):
        return self.triplest(parent_child_features, target)

    def criterion4(self, batch_scores, batch_labels):
        return self.cross_loss(batch_scores, batch_labels)

    def criterion5(self, center_features, labels):
        return self.xenter_loss(center_features, labels), self.optimizer_xentloss, self.xenter_loss

def train_residual_vgg(model, data_loader, criterion, optimizer, device, ccl_weight, bce_weight, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_x1, batch_x2, batch_labels, parent_ids, child_ids) in enumerate(data_loader):

        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)

        batch_labels = batch_labels.to(device)
        gt_all = batch_labels.view(-1, 1).float()

        parent_ids = parent_ids.to(device)
        child_ids = child_ids.to(device)
        image_ids = torch.cat((parent_ids, child_ids), dim=0)  # KI KII # mix two lists of ids
        target = image_ids

        # batch_scores, f_parent, f_child = model(batch_x1, batch_x2)
        batch_scores, f_parent, f_child, family_id_dict, center_feature = model(batch_x1, batch_x2)

        preds = batch_scores > 0.5
        epoch_train_acc += float(torch.sum(preds == (gt_all > 0.5)))
        nb_data += float(gt_all.size(0))


        f = torch.cat((f_parent, f_child), dim=0)

        p1 = (f_parent * gt_all)
        p2 = (f_child * gt_all)
        n1 = f_parent * (1.0 - gt_all)
        n2 = f_child * (1.0 - gt_all)

        # another_gt = batch_labels

        simi = torch.cosine_similarity(f_parent, f_child, dim=1).view(-1, 1)

        # loss1 = criterion.criterion4(batch_scores, batch_labels)
        loss1 = criterion.criterion1(batch_scores, gt_all) # BCEWithLogitsLoss
        loss2 = criterion.criterion2(p1, p2)  # mse loss for Positive sample feature distance /omega1 0.5
        loss3 = criterion.criterion2(n1, n2)  # Negative sample feature distance /omega2 0.1
        loss4 = criterion.criterion2(simi, gt_all)  # cosine similarity
        loss5 = criterion.criterion3(f, target)  # tripletloss: Three yuan loss /alpha 0.5
        loss7, xent_optimizer, xenter_loss = criterion.criterion5(center_feature, gt_all)



        # loss6 = criterion.criterion4(family_id_dict[0], target)
        num_part = 4#2#4#3
        loss6 = criterion.criterion4(family_id_dict[0], target)
        for i in range(num_part - 1):
            loss6 += criterion.criterion4(family_id_dict[i + 1], target)

        J =  bce_weight * (loss1 + 5 * loss2 + 0.6 * loss3  + loss4 + loss5 + loss6) + ccl_weight * loss7

        
        # if epoch > st.KICK_IN_EPOCH:
        #     J =  loss1 + 5 * loss2 + 0.6 * loss3  + loss4 + loss5 + loss6 + ccl_weight * loss7
        #     optimizer.zero_grad()
        #     xent_optimizer.zero_grad()
            
        #     J.backward()
        #     optimizer.step()
            
        #     # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        #     for param in xenter_loss.parameters():
        #         param.grad.data *= (1./ccl_weight)
        #     xent_optimizer.step()
            
        
        # # elif epoch > st.KICK_IN_EPOCH:
        # #     J = bce_weight * (loss1 + 5 * loss2 + 0.6 * loss3  + loss4 + loss5 + loss6)
        # #     optimizer.zero_grad()
        # #     J.backward()
        # #     optimizer.step()
        
        # else:
        #     J = loss1 + 5 * loss2 + 0.6 * loss3  + loss4 + loss5 + loss6
            
        #     optimizer.zero_grad()
        #     J.backward()
        #     optimizer.step()
            


        # print("Center_Loss                  = ", f'{loss7.item():.3f}\t\t{ccl_weight:.3f}')
        # print("BCE_loss                     = ", f'{loss1.item():.5f}\t\t{bce_weight:.3f}')
        # print("MSE_loss (Positive Samples)  = ", f'{loss2.item()*5:.5f}')
        # print("MSE_loss (Negative Samples)  = ", f'{loss3.item()*0.6:.5f}')
        # print("MSE_loss (Cosine similarity) = ", f'{loss4.item():.5f}')
        # print("CE_loss                      = ", f'{loss6.item():.5f}')
        # print("----------------------------------------------\n")

        optimizer.zero_grad()
        xent_optimizer.zero_grad()
        
        J.backward()
        optimizer.step()
        
        # multiple (1./alpha) in order to remove the effect of alpha on updating centers
        for param in xenter_loss.parameters():
            try:
                param.grad.data *= (1./ccl_weight)
            except ZeroDivisionError:
                pass 
        xent_optimizer.step()
        


        epoch_loss += J.detach().item()
        # epoch_train_acc += accuracy(batch_scores, batch_labels)
        # nb_data += batch_labels.size(0)
        #print(f'epoch_loss={epoch_loss}, epoch_train_acc={epoch_train_acc}')

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, loss7, loss1, loss2, loss3, loss4, loss5, loss6

def evaluate_metric(model, data_loader, criterion, device):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_x1, batch_x2, batch_labels, parent_ids, child_ids) in enumerate(data_loader):

            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)

            batch_labels = batch_labels.to(device)
            gt_all = batch_labels.view(-1, 1).float()

            # parent_ids = parent_ids.to(device)
            # child_ids = child_ids.to(device)
            # image_ids = torch.cat((parent_ids, child_ids), dim=0)  # KI KII


            batch_scores, f_parent, f_child, _,_ = model(batch_x1, batch_x2)          #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            preds = batch_scores > 0.5
            epoch_test_acc += float(torch.sum(preds == (gt_all > 0.5)))
            nb_data += float(gt_all.size(0))

            # J = loss(batch_scores, batch_labels)
            J = criterion.criterion1(batch_scores, gt_all)

            epoch_test_loss += J.detach().item()
            # epoch_test_acc += accuracy(batch_scores, batch_labels)
            # nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc

def detailed_evaluate_metric(model, data_loader, loss, device):

    model.eval()
    epoch_test_acc = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    nb_data = 0

    confusion_matrix_dict = {}

    with torch.no_grad():
        for iter, (batch_x1, batch_x2, batch_labels, parent_ids, child_ids) in enumerate(data_loader):

            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)

            batch_labels = batch_labels.to(device)
            gt_all = batch_labels.view(-1, 1).float()

            # parent_ids = parent_ids.to(device)
            # child_ids = child_ids.to(device)
            # image_ids = torch.cat((parent_ids, child_ids), dim=0)  # KI KII


            batch_scores, f_parent, f_child, _,_ = model(batch_x1, batch_x2)          #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            # J = loss(batch_scores, batch_labels)
            # epoch_test_loss += J.detach().item()
            preds = batch_scores > 0.5
            epoch_test_acc += float(torch.sum(preds == (gt_all > 0.5)))
            nb_data += float(gt_all.size(0))


            # epoch_test_acc += accuracy(batch_scores, batch_labels)
            TN, FP,FN, TP  = confusion_matrix(batch_scores, batch_labels)
            total_TN += TN
            total_FP += FP
            total_FN += FN
            total_TP += TP

            # nb_data += batch_labels.size(0)

        # epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    confusion_matrix_dict['epoch_test_acc'] = epoch_test_acc
    confusion_matrix_dict['total_TN'] = total_TN
    confusion_matrix_dict['total_FP'] = total_FP
    confusion_matrix_dict['total_FN'] = total_FN
    confusion_matrix_dict['total_TP'] = total_TP

    return confusion_matrix_dict

def train_metric_0(model, data_loader, criterion, optimizer, device):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_x1, batch_x2, batch_labels, parent_ids, child_ids) in enumerate(data_loader):

        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)

        batch_labels = batch_labels.to(device)
        gt_all = batch_labels.view(-1, 1).float()

        parent_ids = parent_ids.to(device)
        child_ids = child_ids.to(device)
        image_ids = torch.cat((parent_ids, child_ids), dim=0)  # KI KII


        batch_scores, f_parent, f_child = model(batch_x1, batch_x2)

        target = image_ids
        f = torch.cat((f_parent, f_child), dim=0)

        p1 = (f_parent * gt_all)
        p2 = (f_child * gt_all)
        n1 = f_parent * (1.0 - gt_all)
        n2 = f_child * (1.0 - gt_all)

        # another_gt = batch_labels

        simi = torch.cosine_similarity(f_parent, f_child, dim=1).view(-1, 1)

        loss1 = criterion.criterion4(batch_scores, batch_labels)
        loss2 = criterion.criterion2(p1, p2)  # Positive sample feature distance /omega1 0.5
        loss3 = criterion.criterion2(n1, n2)  # Negative sample feature distance /omega2 0.1
        loss4 = criterion.criterion2(simi, gt_all)  # cosine similarity
        # loss5 = criterion.criterion3(f, target)  # Three yuan loss /alpha 0.5

        # J = loss(batch_scores, batch_labels)
        # J = loss1 + (loss6 / 3.0) + loss5 + 0.6 * loss3 + 5 * loss2 + loss4
        J = loss1 + 5 * loss2 + 0.6 * loss3  + loss4
        optimizer.zero_grad()
        J.backward()
        #model.float()
        optimizer.step()


        epoch_loss += J.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        #print(f'epoch_loss={epoch_loss}, epoch_train_acc={epoch_train_acc}')

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc

def evaluate_metric_0(model, data_loader, loss, device):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_x1, batch_x2, batch_labels, parent_ids, child_ids) in enumerate(data_loader):

            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)

            batch_labels = batch_labels.to(device)
            # gt_all = batch_labels.view(-1, 1).float()

            # parent_ids = parent_ids.to(device)
            # child_ids = child_ids.to(device)
            # image_ids = torch.cat((parent_ids, child_ids), dim=0)  # KI KII


            batch_scores, f_parent, f_child = model(batch_x1, batch_x2)          #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            J = loss(batch_scores, batch_labels)

            epoch_test_loss += J.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc

def detailed_evaluate_metric_0(model, data_loader, loss, device):

    model.eval()
    epoch_test_acc = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    nb_data = 0

    confusion_matrix_dict = {}

    with torch.no_grad():
        for iter, (batch_x1, batch_x2, batch_labels, parent_ids, child_ids) in enumerate(data_loader):

            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)

            batch_labels = batch_labels.to(device)
            # gt_all = batch_labels.view(-1, 1).float()

            # parent_ids = parent_ids.to(device)
            # child_ids = child_ids.to(device)
            # image_ids = torch.cat((parent_ids, child_ids), dim=0)  # KI KII


            batch_scores, f_parent, f_child = model(batch_x1, batch_x2)          #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            # J = loss(batch_scores, batch_labels)
            # epoch_test_loss += J.detach().item()

            epoch_test_acc += accuracy(batch_scores, batch_labels)
            TN, FP,FN, TP  = confusion_matrix(batch_scores, batch_labels)
            total_TN += TN
            total_FP += FP
            total_FN += FN
            total_TP += TP

            nb_data += batch_labels.size(0)

        # epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    confusion_matrix_dict['epoch_test_acc'] = epoch_test_acc
    confusion_matrix_dict['total_TN'] = total_TN
    confusion_matrix_dict['total_FP'] = total_FP
    confusion_matrix_dict['total_FN'] = total_FN
    confusion_matrix_dict['total_TP'] = total_TP

    return confusion_matrix_dict

def train_mixture(model, data_loader, loss, optimizer, device):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_x1, batch_x2, batch_labels) in enumerate(data_loader):

        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        batch_labels = batch_labels.to(device)

        batch_scores = model(batch_x1, batch_x2)
        #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        #model.float()
        optimizer.step()


        epoch_loss += J.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        #print(f'epoch_loss={epoch_loss}, epoch_train_acc={epoch_train_acc}')

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc

def evaluate_mixture(model, data_loader, loss, device):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_x1, batch_x2, batch_labels) in enumerate(data_loader):

          batch_x1 = batch_x1.to(device)
          batch_x2 = batch_x2.to(device)
          batch_labels = batch_labels.to(device)

          batch_scores = model(batch_x1, batch_x2)
          #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
          J = loss(batch_scores, batch_labels)

          epoch_test_loss += J.detach().item()
          epoch_test_acc += accuracy(batch_scores, batch_labels)
          nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc

# ## detailed Evaluation
def detailed_evaluate_mixture(model, data_loader, loss, device):

    model.eval()
    epoch_test_acc = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    nb_data = 0

    confusion_matrix_dict = {}

    with torch.no_grad():
        for iter, (batch_x1, batch_x2, batch_labels) in enumerate(data_loader):
          batch_x1 = batch_x1.to(device)
          batch_x2 = batch_x2.to(device)
          batch_labels = batch_labels.to(device)


          batch_scores = model(batch_x1, batch_x2)
          #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
          # J = loss(batch_scores, batch_labels)
          # epoch_test_loss += J.detach().item()

          epoch_test_acc += accuracy(batch_scores, batch_labels)
          TN, FP,FN, TP  = confusion_matrix(batch_scores, batch_labels)
          total_TN += TN
          total_FP += FP
          total_FN += FN
          total_TP += TP

          nb_data += batch_labels.size(0)

        # epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    confusion_matrix_dict['epoch_test_acc'] = epoch_test_acc
    confusion_matrix_dict['total_TN'] = total_TN
    confusion_matrix_dict['total_FP'] = total_FP
    confusion_matrix_dict['total_FN'] = total_FN
    confusion_matrix_dict['total_TP'] = total_TP

    return confusion_matrix_dict


def train_e2e(model, data_loader, loss, optimizer, device):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_x1, batch_x2, batch_labels) in enumerate(data_loader):

        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        batch_labels = batch_labels.to(device)

        batch_scores = model(batch_x1, batch_x2)
        #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        #model.float()
        optimizer.step()


        epoch_loss += J.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        #print(f'epoch_loss={epoch_loss}, epoch_train_acc={epoch_train_acc}')

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc

def evaluate_e2e(model, data_loader, loss, device):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_x1, batch_x2, batch_labels) in enumerate(data_loader):

          batch_x1 = batch_x1.to(device)
          batch_x2 = batch_x2.to(device)
          batch_labels = batch_labels.to(device)

          batch_scores = model(batch_x1, batch_x2)
          #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
          J = loss(batch_scores, batch_labels)

          epoch_test_loss += J.detach().item()
          epoch_test_acc += accuracy(batch_scores, batch_labels)
          nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc

# ## detailed Evaluation
def detailed_evaluate_e2e(model, data_loader, loss, device):

    model.eval()
    epoch_test_acc = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    nb_data = 0

    confusion_matrix_dict = {}

    with torch.no_grad():
        for iter, (batch_x1, batch_x2, batch_labels) in enumerate(data_loader):
          batch_x1 = batch_x1.to(device)
          batch_x2 = batch_x2.to(device)
          batch_labels = batch_labels.to(device)


          batch_scores = model(batch_x1, batch_x2)
          #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
          # J = loss(batch_scores, batch_labels)
          # epoch_test_loss += J.detach().item()

          epoch_test_acc += accuracy(batch_scores, batch_labels)
          TN, FP,FN, TP  = confusion_matrix(batch_scores, batch_labels)
          total_TN += TN
          total_FP += FP
          total_FN += FN
          total_TP += TP

          nb_data += batch_labels.size(0)

        # epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    confusion_matrix_dict['epoch_test_acc'] = epoch_test_acc
    confusion_matrix_dict['total_TN'] = total_TN
    confusion_matrix_dict['total_FP'] = total_FP
    confusion_matrix_dict['total_FN'] = total_FN
    confusion_matrix_dict['total_TP'] = total_TP

    return confusion_matrix_dict

# ## Train one epoch
def train_att(model, data_loader, loss, optimizer):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']
        # batch_E = batch_E[:,None] # reshape batch_E to [:, 1], e.g increase the dimensions to 2 (size of all, 1)
                            # because they must multiply by  (size, 1)*(1,hidden_dim), batch_E *X(1,hidden_dim)


        batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
        #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        #model.float()
        optimizer.step()


        epoch_loss += J.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        #print(f'epoch_loss={epoch_loss}, epoch_train_acc={epoch_train_acc}')

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc

def train(model, data_loader, loss, optimizer, device):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']
        # batch_E = batch_E[:,None] # reshape batch_E to [:, 1], e.g increase the dimensions to 2 (size of all, 1)
                            # because they must multiply by  (size, 1)*(1,hidden_dim), batch_E *X(1,hidden_dim)

        batch_X = batch_X.to(device)
        batch_E = batch_E.to(device)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)

        batch_scores = model(batch_graphs, batch_X, batch_E)
        #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        #model.float()
        optimizer.step()


        epoch_loss += J.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        #print(f'epoch_loss={epoch_loss}, epoch_train_acc={epoch_train_acc}')

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc

# ## Train one epoch
def train_simple_data(model, data_loader, loss, optimizer, Device):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_samples, batch_labels) in enumerate(data_loader):

        batch_samples = batch_samples.to(Device)
        batch_labels  = batch_labels.to(Device)
        batch_scores = model(batch_samples)
#         print(f'batch_scores={batch_scores.shape}, batch_labels={batch_labels.shape},\
#                 batch_scores_type={(batch_scores.type())}, batch_labels_type={(batch_labels.type())}')
#               batch_labels_unsqueeze={batch_labels.unsqueeze(1).shape}')

        #J = loss(batch_scores, batch_labels.unsqueeze(1))
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        #model.float()
        optimizer.step()


        epoch_loss += J.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        #print(f'epoch_loss={epoch_loss}, epoch_train_acc={epoch_train_acc}')

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc


# ## Evaluation
def evaluate(model, data_loader, loss, device):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_X = batch_graphs.ndata['feat']
            batch_E = batch_graphs.edata['feat']
            # batch_E = batch_E[:,None] # reshape batch_E to [:, 1], e.g increase the dimensions to 2 (size of all, 1)
                            # because they must multiply by  (size, 1)*(1,hidden_dim), batch_E *X(1,hidden_dim)

            batch_X = batch_X.to(device)
            batch_E = batch_E.to(device)
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model(batch_graphs, batch_X, batch_E)
            #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            J = loss(batch_scores, batch_labels)

            epoch_test_loss += J.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc

# ## detailed Evaluation
def detailed_evaluate(model, data_loader, loss, device):

    model.eval()
    epoch_test_acc = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    nb_data = 0

    confusion_matrix_dict = {}

    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_X = batch_graphs.ndata['feat']
            batch_E = batch_graphs.edata['feat']
            # batch_E = batch_E[:,None] # reshape batch_E to [:, 1], e.g increase the dimensions to 2 (size of all, 1)
                    # because they must multiply by  (size, 1)*(1,hidden_dim), batch_E *X(1,hidden_dim)

            batch_X = batch_X.to(device)
            batch_E = batch_E.to(device)
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_labels.to(device)


            batch_scores = model(batch_graphs, batch_X, batch_E)
            #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            # J = loss(batch_scores, batch_labels)
            # epoch_test_loss += J.detach().item()

            epoch_test_acc += accuracy(batch_scores, batch_labels)
            TN, FP,FN, TP  = confusion_matrix(batch_scores, batch_labels)
            total_TN += TN
            total_FP += FP
            total_FN += FN
            total_TP += TP

            nb_data += batch_labels.size(0)

        # epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    confusion_matrix_dict['epoch_test_acc'] = epoch_test_acc
    confusion_matrix_dict['total_TN'] = total_TN
    confusion_matrix_dict['total_FP'] = total_FP
    confusion_matrix_dict['total_FN'] = total_FN
    confusion_matrix_dict['total_TP'] = total_TP

    return confusion_matrix_dict

def evaluate_simple_data(model, data_loader, loss, Device):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_samples, batch_labels) in enumerate(data_loader):

            batch_samples = batch_samples.to(Device)
            batch_labels = batch_labels.to(Device)

            batch_scores = model(batch_samples)
            #print(f'batch_scores={batch_scores}, batch_labels={batch_labels}')
            #J = loss(batch_scores, batch_labels.unsqueeze(1))
            J = loss(batch_scores, batch_labels)

            epoch_test_loss += J.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc

def create_trn_tst_graphset(graphset, fold_num):

    folds = graphset.graph_folds#.tolist()
    # tst_indices = folds.index( (folds == fold_num))
    tst_indices =  [idx for idx, value in enumerate(folds == fold_num) if value]
    trn_indices =  [idx for idx, value in enumerate(folds != fold_num) if value]

    # print(tst_indices, trn_indices)

    return Subset(graphset, trn_indices),  Subset(graphset, tst_indices)

def convert_float2str(num):
    # return np.format_float_positional(np.float16(num))
    return '%.2f'%num
