import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, num_class, gamma=2, alpha=None):
        '''
        alpha: tensor of shape (C)
        '''
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_class = num_class
        if alpha==None:
            self.alpha = torch.ones(num_class)
        if isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class)
            self.alpha = alpha / alpha.sum()

    def forward(self, logit, target):
        '''
        args: logits: tensor before the softmax of shape (N,C) where C = number of classes 
            or (N, C, H, W) in case of 2D Loss, 
            or (N,C,d1,d2,...,dK) where K≥1 in the case of K-dimensional loss.
        args: label: (N) where each value is in [0,C-1],
            or (N, d1, d2, ..., dK)
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
        '''
        if self.alpha.device != logit.device:
            self.alpha = self.alpha.to(logit.device)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)#(N,C,d=d1*d2*d3)
            logit = logit.permute(0,2,1)#(N,d,C)
            logit = logit.view(-1, self.num_class) #(N*d,C)
        target = target.view(-1) #(N*H*W)
        #alpha  = self.alpha.view(1, self.num_class) #(1,C)
        alpha = self.alpha[target.cpu().long()] #(N*H*W)

        logpt = - F.cross_entropy(logit, target, reduction='none')
        pt    = torch.exp(logpt)
        focal_loss = -(alpha * (1 - pt) ** self.gamma) * logpt

        return focal_loss.mean()

'''
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            select = (target!=0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0, select.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
'''