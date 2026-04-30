import torch
import torch.nn as nn
import torch.nn.functional as F

def lm_loss(logits,targets):
    B,T,V = logits.shape
    logits = logits.view(B*T,V)
    targets = targets.view(B*T)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits,targets)

    return loss


def contrastive_loss(image_feat,text_feat,temperature=0.07):
    image_feat = F.normalize(image_feat,dim=-1)
    text_feat = F.normalize(text_feat,dim=-1)
    logits = image_feat @ text_feat.T / temperature
    labels = torch.arange(image_feat.size(0)).to(image_feat.device)
    loss_i2t = F.cross_entropy(logits,labels)
    loss_t2i = F.cross_entropy(logits,labels)

    return (loss_i2t + loss_t2i) / 2


def matching_loss(logits,labels):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits,labels)