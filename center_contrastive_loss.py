import torch
import torch.nn as nn
import settings as st

class CenterContrastiveLoss(nn.Module):
    """Center Contrastive Loss.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        margin (float): margin for the contrastive loss.
        use_gpu (bool): whether to use GPU.
    """
    def __init__(self, num_classes=10, feat_dim=st.config[st.I]['FC1'], margin=st.config[st.I]['FC1']*2, use_gpu=True):
        super(CenterContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        # Compute the distance matrix between features and centers
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        # Create a mask for the same class
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # Calculate intra-class distances
        intra_class_dist = distmat * mask.float()
        intra_class_loss = intra_class_dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        # Calculate inter-class distances
        mask_neg = 1 - mask.float()
        inter_class_dist = distmat * mask_neg
        inter_class_dist = inter_class_dist[inter_class_dist != 0]
        inter_class_dist = inter_class_dist.view(-1)
        inter_class_dist = inter_class_dist[inter_class_dist < self.margin]
        #print(inter_class_dist)
        #exit(0)
        # print(max(inter_class_dist), min(inter_class_dist), sum(inter_class_dist)/len(inter_class_dist))
        inter_class_loss = torch.clamp(inter_class_dist, min=0).sum() / batch_size
        # Total loss
        total_loss = intra_class_loss + inter_class_loss

        # print(f'{intra_class_loss.item():.3f}','\t', f'{inter_class_loss.item():.3f}')
        # print(f'{(sum(self.centers[0])/len(self.centers[0])).item():.5f}')
        # print(f'{(sum(self.centers[1])/len(self.centers[1])).item():.5f}\n')
        
        return total_loss