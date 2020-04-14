import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch as th


class MaxMarginRankingLoss(nn.Module):
    def __init__(self, margin=1):
        super(MaxMarginRankingLoss, self).__init__()
        # self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, conf_matrix, gt_idx):
        # calculates "uni"-directional loss from vid to text
        n = conf_matrix.size()  # TODO: should be batch_size x num_uniq_text #n[0] must be batch size

        assert len(conf_matrix) == len(gt_idx)
        row_idx = np.arange(n[0])
        sii = conf_matrix[row_idx,gt_idx]

        sii = sii.unsqueeze(1)  # TODO: dims should be batch_size x 1
        sii = sii.expand(n)  # TODO: dims should be batch_size x num_uniq_text # must have dims of conf_matrix
        sii = sii.contiguous().view(-1, 1)

        sij = conf_matrix.view(-1, 1)

        max_margin = F.relu(self.margin - (sii - sij))

        return max_margin.sum() / n[0]  # TODO: DOUBT: is there a mistake in author's last statement (.mean())
        # should be divided only by batch size but using mean, the sum will be divided by count of all elements in max_margin
