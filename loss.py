import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def top_k_accuracy(logits, labels, ks=[1, 5]):
    """
    计算给定 logits 和 labels 的 top-k 准确率。
    
    参数：
    - logits: Tensor of shape (N, num_classes), 模型的预测输出
    - labels: Tensor of shape (N,), 包含实际类别的下标格式的标签
    - ks: List of integers, 指定要计算的 k 值，如 [1, 5] 表示计算 top-1 和 top-5 准确率
    
    返回：
    - acc_list: List of accuracies corresponding to each k in ks
    """
    max_k = max(ks)  # 计算需要的最大 k 值
    batch_size = labels.size(0)

    # 获取 top-k 预测，返回的 top_preds 形状为 (N, max_k)
    _, top_preds = logits.topk(max_k, dim=1, largest=True, sorted=True)

    # 扩展 labels 的形状以便与 top_preds 比较，形状为 (N, max_k)
    top_preds = top_preds.t()  # 转置为 (max_k, N)
    correct = top_preds.eq(labels.view(1, -1).expand_as(top_preds))  # 检查 top-k 内是否有正确的标签

    # 计算每个 k 对应的准确率
    acc_list = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # 获取前 k 的正确预测数量
        accuracy = correct_k.mul_(100.0 / batch_size).item()  # 计算准确率百分比
        acc_list.append(accuracy)

    return acc_list
class SimCLRLoss(nn.Module):

    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.CEL = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to(self, device):
        self.device = device
        self.CEL = self.CEL.to(device)
        return self

    def info_nce_loss(self, features):
        
        device = self.device

        # print(features.shape)
        bs = int(features.shape[0] // 2)
        labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Put the positive column at the end (when all the entries are the same, the top1 acc will be 0; while if the
        # positive column is at the start, the top1 acc might be exaggerated)
        logits = torch.cat([negatives, positives], dim=1)
        # The label means the last column contain the positive pairs
        labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
        labels = labels.to(device)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, features):
        features = stratified_layerNorm(features, features.shape[0] // 2)
        # fea need to be normalized to 1
        features = F.normalize(features, dim=1)
        self.to(features.device)
        logits, labels = self.info_nce_loss(features)
        loss = self.CEL(logits, labels)
        [acc_1, acc_5] = top_k_accuracy(logits, labels)
        return loss, logits, labels, [acc_1, acc_5]

def stratified_layerNorm(out, n_samples):
    n_samples = int(n_samples)
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1)
        out_oneSub_str = out_oneSub.clone()
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-5)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str
    return out_str