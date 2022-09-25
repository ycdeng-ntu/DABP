import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add, scatter_softmax, scatter_mean

from alg.constant import VAR_ID, V2F_ID, F2V_ID, device

torch.autograd.set_detect_anomaly(True)


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.query_proj = nn.Linear(hidden, hidden, bias=False)
        self.key_proj = nn.Linear(hidden, hidden, bias=False)
        self.score_proj = nn.Linear(2 * hidden, 1)

    def forward(self, query, key):
        query = self.query_proj(query)
        key = self.key_proj(key)
        return torch.sigmoid(self.score_proj(torch.cat([query, key], dim=1)))


class AttentiveBP(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, prefix_dim=4, msg_dim=15, max_color=100):
        super().__init__()
        self.prefix_dim = prefix_dim
        self.vn_color_embed = nn.Embedding(max_color, in_channels - prefix_dim)
        self.gru1 = nn.GRUCell(msg_dim, in_channels - prefix_dim)
        self.gru2 = nn.GRUCell(msg_dim, in_channels - prefix_dim)
        self.conv1 = GATConv(in_channels, 8, heads=4, concat=True)
        self.conv2 = GATConv(32, 8, 4, concat=True)
        self.conv3 = GATConv(32, 8, 4, concat=True)
        self.conv4 = GATConv(32, out_channels, 4, concat=False)
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(Attention(out_channels))

    def step(self, *param):
        edge_index = param[0]
        var_embed = param[1]
        func_embed = param[2]
        msg_hidden = param[3]
        msgs = param[4]
        msg_rv2f_idxes = param[5]
        msg_cv2f_idxes = param[6]
        msg_f2rv_idxes = param[7]
        msg_f2cv_idxes = param[8]
        msg_v2f_idxes = param[9]
        msg_f2v_idxes = param[10]
        msg_trg_idxes = param[11]
        msg_src_idxes = param[12]
        msg_f2v_per_v_idxes = param[13]
        embed_trg_idxes = param[14]
        embed_src_idxes = param[15]
        v2f_scatter_idxes = param[16]
        f2v_per_v_scatter_idxes = param[17]
        f_batch = param[18]
        cost_tensors = param[19]
        rv_idxes = param[20]
        cv_idxes = param[21]
        first_iteration = param[22]

        # msg f -> v
        msg_rv2f = msgs[msg_rv2f_idxes]
        msg_cv2f = msgs[msg_cv2f_idxes]
        msg_f2rv = torch.min(cost_tensors + msg_cv2f.unsqueeze(1), dim=2)[0]
        msg_f2cv = torch.min(cost_tensors + msg_rv2f.unsqueeze(2), dim=1)[0]
        msgs[msg_f2rv_idxes] = msg_f2rv
        msgs[msg_f2cv_idxes] = msg_f2cv

        _var_embed = []
        for ve in var_embed:
            _var_embed.append(self.vn_color_embed(ve))

        for i in range(len(msg_hidden)):
            v2f_hidden, f2v_hidden = msg_hidden[i]
            v2f_msgs = msgs[msg_v2f_idxes[i]]
            f2v_msgs = msgs[msg_f2v_idxes[i]]
            v2f_hidden = self.gru1(v2f_msgs.detach(), v2f_hidden)
            f2v_hidden = self.gru2(f2v_msgs.detach(), f2v_hidden)
            msg_hidden[i] = [v2f_hidden, f2v_hidden]

        x = []
        for i in range(len(msg_hidden)):
            ve = _var_embed[i]
            pref = torch.Tensor(VAR_ID).double().to(device)
            pref = pref.repeat(ve.shape[0], 1)
            ve = torch.cat([pref, ve], dim=1)

            fe = func_embed[i]

            v2f_hidden = msg_hidden[i][0]
            pref = torch.Tensor(V2F_ID).double().to(device)
            pref = pref.repeat(v2f_hidden.shape[0], 1)
            v2f_hidden = torch.cat([pref, v2f_hidden], dim=1)

            f2v_hidden = msg_hidden[i][1]
            pref = torch.Tensor(F2V_ID).double().to(device)
            pref = pref.repeat(f2v_hidden.shape[0], 1)
            f2v_hidden = torch.cat([pref, f2v_hidden], dim=1)

            x.append(torch.cat([ve, fe, v2f_hidden, f2v_hidden], dim=0))
        x = torch.cat(x, dim=0)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        msg_src = msgs[msg_src_idxes]
        msg_trg = msgs[msg_trg_idxes]

        embed_src = x[embed_src_idxes]
        embed_trg = x[embed_trg_idxes]
        var_degree = torch.unique(v2f_scatter_idxes, return_counts=True)[-1]  # Degree(x) - 1
        embed_trg_repeated = torch.repeat_interleave(embed_trg, var_degree, dim=0)
        assert embed_src.shape == embed_trg_repeated.shape

        attention_scores = []
        attention_trg_scores = []
        for m in self.attentions:
            attention_scores.append(m(embed_trg_repeated, embed_src))
            attention_trg_scores.append(m(embed_trg, embed_trg))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_trg_scores = torch.cat(attention_trg_scores, dim=1)
        attention_weight = scatter_softmax(attention_scores, v2f_scatter_idxes, dim=0)
        weighted_msg = msg_src.unsqueeze(2) * attention_weight.unsqueeze(1)
        weighted_msg = scatter_add(weighted_msg, v2f_scatter_idxes, dim=0)
        weighted_msg = weighted_msg.mean(-1)
        weighted_msg = weighted_msg * var_degree.unsqueeze(1)

        attention_scores = scatter_mean(attention_scores, v2f_scatter_idxes, dim=0)
        damped_weights = torch.softmax(
            torch.cat([attention_scores.unsqueeze(1), attention_trg_scores.unsqueeze(1)], dim=1), dim=1)
        v2f_msgs = weighted_msg.unsqueeze(2) * damped_weights[:, 0, :].unsqueeze(1) + msg_trg.unsqueeze(
            2) * damped_weights[:, 1, :].unsqueeze(1)
        v2f_msgs = v2f_msgs.mean(-1)
        v2f_msgs = v2f_msgs - v2f_msgs.min(dim=1, keepdim=True)[0]
        msgs[msg_trg_idxes] = v2f_msgs

        f2v_per_v_msgs = msgs[msg_f2v_per_v_idxes]
        belief = scatter_add(f2v_per_v_msgs, f2v_per_v_scatter_idxes, dim=0)
        dist = torch.softmax(-belief, dim=1)
        entropy = dist * torch.log2(dist + 1e-6)
        entropy = -entropy.sum(dim=1)
        entropy = entropy.mean()
        dist_rv = dist[rv_idxes]
        dist_cv = dist[cv_idxes]
        if not first_iteration:
            loss = torch.bmm(torch.bmm(dist_rv.unsqueeze(1), cost_tensors), dist_cv.unsqueeze(2)).squeeze()
            loss = scatter_add(loss, f_batch)
            loss = loss.mean()
            loss = loss + 0.1 * entropy
        else:
            loss = 0
        val_idx_rv = dist_rv.argmax(dim=1, keepdim=False).tolist()
        val_idx_cv = dist_cv.argmax(dim=1, keepdim=False).tolist()
        cost = []
        for i in range(cost_tensors.shape[0]):
            cost.append(cost_tensors[i, val_idx_rv[i], val_idx_cv[i]].item())
        cost = torch.Tensor(cost).to(device)
        cost = scatter_add(cost, f_batch, dim=0)
        return loss, cost.mean().item()

    def preprocess(self, data):
        bs = data.num_graphs
        sizes = data.sizes  # [2] * bs

        edge_index = data.edge_index.clone().to(device)  # (bs * 8 * NF, 2)

        var_embed = list(data.var_embed)  # [(1, in_channel)] * bs
        for i in range(bs):
            var_embed[i] = torch.Tensor(var_embed[i]).long().to(device)

        func_embed = list(data.func_embed)  # [(NF, in_channels)] * bs
        for i in range(bs):
            func_embed[i] = torch.Tensor(func_embed[i]).double().to(device)

        msg_hidden = copy.deepcopy(data.msg_hidden)  # [[(2 * NF, in_channel - prefix_dim)] * 2] * bs
        for i in range(bs):
            for j in range(2):
                msg_hidden[i][j] = torch.Tensor(msg_hidden[i][j]).double().to(device)

        msgs = data.msgs.clone().to(device)  # (4 * NF * bs, dom_size)

        msg_rv2f_idxes = list(data.msg_rv2f_idxes)  # [(NF,)] * bs
        msg_cv2f_idxes = list(data.msg_cv2f_idxes)  # [(NF,)] * bs
        msg_f2rv_idxes = list(data.msg_f2rv_idxes)  # [(NF,)] * bs
        msg_f2cv_idxes = list(data.msg_f2cv_idxes)  # [(NF,)] * bs
        msg_v2f_idxes = list(data.msg_v2f_idxes)  # [(2 * NF,)] * bs
        msg_f2v_idxes = list(data.msg_f2v_idxes)  # [(2 * NF,)] * bs
        msg_trg_idxes = list(data.msg_trg_idxes)  # [(<=2 * NF,)] * bs
        msg_src_idxes = list(data.msg_src_idxes)  # [(?,)] * bs
        msg_f2v_per_v_idxes = list(data.msg_f2v_per_v_idxes)  # [(2 * NF,)] * bs
        offset = 0
        for i in range(bs):
            msg_rv2f_idxes[i] = torch.Tensor(msg_rv2f_idxes[i]).long().to(device) + offset
            msg_cv2f_idxes[i] = torch.Tensor(msg_cv2f_idxes[i]).long().to(device) + offset
            msg_f2rv_idxes[i] = torch.Tensor(msg_f2rv_idxes[i]).long().to(device) + offset
            msg_f2cv_idxes[i] = torch.Tensor(msg_f2cv_idxes[i]).long().to(device) + offset
            msg_v2f_idxes[i] = torch.Tensor(msg_v2f_idxes[i]).long().to(device) + offset
            msg_f2v_idxes[i] = torch.Tensor(msg_f2v_idxes[i]).long().to(device) + offset
            msg_trg_idxes[i] = torch.Tensor(msg_trg_idxes[i]).long().to(device) + offset
            msg_src_idxes[i] = torch.Tensor(msg_src_idxes[i]).long().to(device) + offset
            msg_f2v_per_v_idxes[i] = torch.Tensor(msg_f2v_per_v_idxes[i]).long().to(device) + offset
            offset += 4 * sizes[i][-1]
        msg_f2v_per_v_idxes = torch.cat(msg_f2v_per_v_idxes)  # (2 * NF * bs,)
        msg_rv2f_idxes = torch.cat(msg_rv2f_idxes)
        msg_cv2f_idxes = torch.cat(msg_cv2f_idxes)
        msg_f2rv_idxes = torch.cat(msg_f2rv_idxes)
        msg_f2cv_idxes = torch.cat(msg_f2cv_idxes)
        msg_trg_idxes = torch.cat(msg_trg_idxes)
        msg_src_idxes = torch.cat(msg_src_idxes)

        assert torch.unique(msg_f2v_per_v_idxes).numel() == 2 * sum([k[-1] for k in sizes])

        embed_trg_idxes = list(data.embed_trg_idxes)  # [(<=2 * NF,)] * bs
        embed_src_idxes = list(data.embed_src_idxes)  # [(?,)] * bs
        offset = 0
        for i in range(bs):
            embed_trg_idxes[i] = torch.Tensor(embed_trg_idxes[i]).long().to(device) + offset
            embed_src_idxes[i] = torch.Tensor(embed_src_idxes[i]).long().to(device) + offset
            offset += data.get_example(i).x.shape[0]
        embed_trg_idxes = torch.cat(embed_trg_idxes)  # (<=2 * NF * bs,)
        embed_src_idxes = torch.cat(embed_src_idxes)  # (? * bs,)

        v2f_scatter_idxes = list(data.v2f_scatter_idxes)  # [(?,)] * bs
        offset = 0
        for i in range(bs):
            tmp = v2f_scatter_idxes[i][-1] + 1
            assert tmp <= 2 * sizes[i][-1]
            v2f_scatter_idxes[i] = torch.Tensor(v2f_scatter_idxes[i]).long().to(device) + offset
            offset += tmp
        v2f_scatter_idxes = torch.cat(v2f_scatter_idxes)  # (? * bs,)

        f2v_per_v_scatter_idxes = list(data.f2v_per_v_scatter_idxes)  # [(2 * NF,)] * bs
        offset = 0
        for i in range(bs):
            tmp = f2v_per_v_scatter_idxes[i][-1] + 1
            assert tmp == sizes[i][0]
            f2v_per_v_scatter_idxes[i] = torch.Tensor(f2v_per_v_scatter_idxes[i]).long().to(device) + offset
            offset += tmp
        f2v_per_v_scatter_idxes = torch.cat(f2v_per_v_scatter_idxes)  # (2 * NF * bs)

        f_batch = list(data.f_batch)
        for i in range(bs):
            f_batch[i] = torch.Tensor(f_batch[i]).long().to(device) * i
        f_batch = torch.cat(f_batch)

        cost_tensors = data.cost_tensors.double().to(device)

        rv_idxes = list(data.rv_idxes)
        cv_idxes = list(data.cv_idxes)
        offset = 0
        for i in range(bs):
            rv_idxes[i] = torch.Tensor(rv_idxes[i]).long().to(device) + offset
            cv_idxes[i] = torch.Tensor(cv_idxes[i]).long().to(device) + offset
            offset += sizes[i][0]
        rv_idxes = torch.cat(rv_idxes)
        cv_idxes = torch.cat(cv_idxes)

        self.edge_index = edge_index
        self.var_embed = var_embed
        self.func_embed = func_embed
        self.msg_hidden = msg_hidden
        self.msgs = msgs
        self.msg_rv2f_idxes = msg_rv2f_idxes
        self.msg_cv2f_idxes = msg_cv2f_idxes
        self.msg_f2rv_idxes = msg_f2rv_idxes
        self.msg_f2cv_idxes = msg_f2cv_idxes
        self.msg_v2f_idxes = msg_v2f_idxes
        self.msg_f2v_idxes = msg_f2v_idxes
        self.msg_trg_idxes = msg_trg_idxes
        self.msg_src_idxes = msg_src_idxes
        self.msg_f2v_per_v_idxes = msg_f2v_per_v_idxes
        self.embed_trg_idxes = embed_trg_idxes
        self.embed_src_idxes = embed_src_idxes
        self.v2f_scatter_idxes = v2f_scatter_idxes
        self.f2v_per_v_scatter_idxes = f2v_per_v_scatter_idxes
        self.f_batch = f_batch
        self.cost_tensors = cost_tensors
        self.rv_idxes = rv_idxes
        self.cv_idxes = cv_idxes

    def forward(self, n_iter=150):
        losses = []
        costs = []
        for i in range(len(self.msg_hidden)):
            for j in range(2):
                self.msg_hidden[i][j] = self.msg_hidden[i][j].detach()
        self.msgs = self.msgs.detach()

        for i in range(n_iter):
            l, c = self.step(self.edge_index, self.var_embed, self.func_embed, self.msg_hidden, self.msgs,
                             self.msg_rv2f_idxes, self.msg_cv2f_idxes, self.msg_f2rv_idxes, self.msg_f2cv_idxes,
                             self.msg_v2f_idxes, self.msg_f2v_idxes, self.msg_trg_idxes, self.msg_src_idxes,
                             self.msg_f2v_per_v_idxes, self.embed_trg_idxes, self.embed_src_idxes, self.v2f_scatter_idxes,
                             self.f2v_per_v_scatter_idxes, self.f_batch, self.cost_tensors,
                             self.rv_idxes, self.cv_idxes, i == 0)
            if i != 0:
                losses.append(l)
                costs.append(c)
        return losses, costs
