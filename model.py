import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, mod_dim, text_dim, latent_dim):
        """
        mod_dim: dictionary containing dims of different modalities/sources
        txt_dim: dim of word2vec
        latent_dim: dim of latent space
        """
        super(Net, self).__init__()
        self.mee = MEE(mod_dim, text_dim, latent_dim)

    def forward(self, mod, ind, text_uniq):
        return self.mee(mod, ind, text_uniq)


class MEE(nn.Module):

    def __init__(self, mod_dim, text_dim, latent_dim):
        super(MEE, self).__init__()

        m = list(mod_dim.keys())
        self.m = m

        self.modalities_GU = nn.ModuleList([
            Gated_Embedding_Unit(mod_dim[m[i]], latent_dim, gating=False) for i in range(len(m))
        ])
        self.text_GU = Gated_Embedding_Unit(text_dim, latent_dim, gating=False)

    def forward(self, mod, ind, text_uniq):
        """
        mod -> dictionary containing input from each modalitiy. each keys contains val of dim: batch_size*mod_dim
        text_uniq -> contains word2vec of ALL available txt in dataset: num_uniq_text x text_dim

        returns:
            sim_matrix: matrix[i][j] contains aggregated similairty between "i"th text and "j"th frame/segment/input of batch (unweighted mean over all modalities)
        """

        for i, l in enumerate(self.modalities_GU):
            mod[self.m[i]] = l(mod[self.m[i]])
        text_uniq_embd = self.text_GU(text_uniq)

        text_uniq_count = len(text_uniq)
        assert len(text_uniq) == len(text_uniq_embd)

        mod_count = len(ind) == len(mod)
        assert len(self.m) == mod_count

        bs = len(ind[self.m[0]])
        for src in self.m:
            assert bs == len(ind[src]) == len(mod[src])

        available_m = np.zeros([bs, mod_count])
        for i, src in enumerate(self.m):
            available_m[:, i] = ind[src]
        available_m = th.from_numpy(available_m).to(device)

        moe_weights = th.ones([text_uniq_count, mod_count]).to(device) * (1 / mod_count)  # potential bug

        moe_weights = available_m[None, :, :] * moe_weights[:, None, :]
        norm_weights = th.sum(moe_weights, dim=2)
        norm_weights = norm_weights.unsqueeze(2)
        moe_weights = th.div(moe_weights, norm_weights)

        sim_matrix = th.zeros(text_uniq_count, bs)
        for i, src in enumerate(self.m):
            mod[src] = mod[src].transpose(0, 1)
            sim_matrix += moe_weights[:, :, i] * th.matmul(text_uniq_embd, mod[src])  # potential bug
        return sim_matrix  # should be text_uniq x BS


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        if gating:
            self.cg = Context_Gating(output_dimension)
        self.gating = gating
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        if self.gating:
            x = self.cg(x)
        x = F.normalize(x)
        # x = self.relu(x)

        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)

        x = th.cat((x, x1), 1)

        return F.glu(x, 1)
