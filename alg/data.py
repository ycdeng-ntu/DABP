import os
from os import path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from core.parser import parse
from alg.constant import MAX_COLOR_NUM, FUN_ID, scale, SPLIT_RATIO


def label_variables(adj_list):
    ordered_keys = sorted([x for x in adj_list.keys()])
    labels = dict()
    for vn in ordered_keys:
        neighbor_vn_name = adj_list[vn]
        all_colors = set()
        for nb in neighbor_vn_name:
            if nb in labels:
                all_colors.add(labels[nb])
        for color in range(MAX_COLOR_NUM):
            if color not in all_colors:
                break
        labels[vn] = color
    return labels


class COP_Dataset(Dataset):
    def __init__(self, root_dir, node_embed_dim=12, max_dom_size=15):
        self.root_dir = root_dir
        self.node_embed_dim = node_embed_dim
        self.max_dom_size = max_dom_size
        self.inst_list = []

        for f in os.listdir(root_dir):
            if f.endswith('.xml'):
                self.inst_list.append(os.path.join(root_dir, f))
        sorted(self.inst_list)

    def __len__(self):
        return len(self.inst_list)

    def __getitem__(self, fn_idx):
        pth = self.inst_list[fn_idx]
        if not path.exists(pth):
            raise ValueError('path not exist.')
        all_vars, all_matrix = parse(pth, scale=scale)

        scfg = []
        for m, row, col in all_matrix:
            m1 = []
            m2 = []
            for r in m:
                m1.append([x * SPLIT_RATIO for x in r])
                m2.append([x * (1 - SPLIT_RATIO) for x in r])
            scfg.append((m1, row, col))
            scfg.append((m2, row, col))
        all_matrix = scfg

        NV = len(all_vars)
        NF = len(all_matrix)

        # ------------create adj list & ordered vars --------------
        adj_list = dict()                   # dict of list; key: vn, value: list of neighboring vn; len: NV
        var_index = dict()                  # dict; key: vn, value: index of vn; len: NV
        tmp = []
        adj_func_list = dict()              # dict of list; key: vn, value: list of its neighboring function indexes; len: NV
        rv_idxes = []                       # list; row vn indexes; len: NV
        cv_idxes = []                       # list; col vn indexes; len: NV
        for idx, vn in enumerate(all_vars):
            vn, _ = vn
            adj_list[vn] = list()
            adj_func_list[vn] = list()
            var_index[vn] = idx
            tmp.append(vn)

        for idx, fn in enumerate(all_matrix):
            _, row, col = fn

            assert idx not in adj_func_list[row]
            assert idx not in adj_func_list[col]
            assert row != col

            if row in adj_list[col]:
                assert col in adj_list[row]
            else:
                adj_list[row].append(col)
                adj_list[col].append(row)

            adj_func_list[row].append(idx)
            adj_func_list[col].append(idx)

            rv_idxes.append(var_index[row])
            cv_idxes.append(var_index[col])

        assert max(rv_idxes) <= NV - 1
        assert min(rv_idxes) >= 0
        assert max(cv_idxes) <= NV - 1
        assert min(cv_idxes) >= 0
        assert len(set(rv_idxes + cv_idxes)) == NV

        all_vars = tmp                                         # list; ordered vn; len: NV

        vn_color_dict = label_variables(adj_list)              # dict: key: vn, val: color

        var_embed = []                                         # list; len: NV
        for vn in all_vars:
            var_embed.append(vn_color_dict[vn])

        padding = [0 for _ in range(self.node_embed_dim - 4)]
        func_embed = []                                        # list; len: NF
        for _ in range(NF):
            func_embed.append(FUN_ID + padding)

        msg_hidden = []                                       # list of list of list; message hidden for RNNs, see below for details; size: (2, 2*NF, ?), ? = node_embed_dim - 4
        msgs = []                                             # list of list; all BP messages; size: (4*NF, max_dom_size)
        msg_rv2f_idxes = []                                   # list; index of row vn -> fn in msgs; len: NF
        msg_cv2f_idxes = []                                   # list; index of col vn -> fn in msgs; len: NF
        msg_f2rv_idxes = []                                   # list; index of fn -> row vn in msgs; len: NF
        msg_f2cv_idxes = []                                   # list; index of fn -> col vn in msgs; len: NF
        i = 0
        msg_hidden.append([])                                 # message hidden for vn -> fn
        for _ in range(NF):
            msg_hidden[-1].append(padding)                    # rv->f
            msgs.append([0] * self.max_dom_size)
            msg_rv2f_idxes.append(i)
            i += 1

        for _ in range(NF):
            msg_hidden[-1].append(padding)                    # cv->f
            msgs.append([0] * self.max_dom_size)
            msg_cv2f_idxes.append(i)
            i += 1

        msg_hidden.append([])                                 # message hidden for fn -> vn
        for _ in range(NF):
            msg_hidden[-1].append(padding)                    # f->rv
            msgs.append([0] * self.max_dom_size)
            msg_f2rv_idxes.append(i)
            i += 1

        for _ in range(NF):
            msg_hidden[-1].append(padding)                    # f->cv
            msgs.append([0] * self.max_dom_size)
            msg_f2cv_idxes.append(i)
            i += 1
        msg_v2f_idxes = msg_rv2f_idxes + msg_cv2f_idxes
        msg_f2v_idxes = msg_f2rv_idxes + msg_f2cv_idxes
        msgs = torch.Tensor(msgs).double()
        x = torch.zeros(NV + 5 * NF, 1)                       # we only use this fake x to correct edge_index when batching
        assert x.shape[0] == len(var_embed) + len(func_embed) + len(msg_hidden[0]) + len(msg_hidden[1])
        assert msgs.shape[0] == len(msg_hidden[0]) + len(msg_hidden[1])

        x_f_start_idx = NV
        x_rv2f_start_idx = NV + NF
        x_cv2f_start_idx = NV + 2 * NF
        x_f2rv_start_idx = NV + 3 * NF
        x_f2cv_start_idx = NV + 4 * NF

        src = []
        dst = []

        for i, fn in enumerate(all_matrix):
            _, row, col = fn
            row_idx = var_index[row]
            col_idx = var_index[col]

            assert row_idx < x_f_start_idx
            assert col_idx < x_f_start_idx
            assert i + x_f_start_idx < x_rv2f_start_idx
            assert i + x_rv2f_start_idx < x_cv2f_start_idx
            assert i + x_cv2f_start_idx < x_f2rv_start_idx
            assert i + x_f2rv_start_idx < x_f2cv_start_idx
            assert i + x_f2cv_start_idx < x.shape[0]

            # rv -> m -> f
            src.append(row_idx)
            dst.append(i + x_rv2f_start_idx)
            src.append(i + x_rv2f_start_idx)
            dst.append(i + x_f_start_idx)

            # cv -> m -> f
            src.append(col_idx)
            dst.append(i + x_cv2f_start_idx)
            src.append(i + x_cv2f_start_idx)
            dst.append(i + x_f_start_idx)

            # f -> m -> rv
            src.append(i + x_f_start_idx)
            dst.append(i + x_f2rv_start_idx)
            src.append(i + x_f2rv_start_idx)
            dst.append(row_idx)

            # f -> m -> cv
            src.append(i + x_f_start_idx)
            dst.append(i + x_f2cv_start_idx)
            src.append(i + x_f2cv_start_idx)
            dst.append(col_idx)

        edge_index = torch.Tensor([src, dst]).long()

        msg_trg_idxes = list()      # list; index of each vn -> fn in msgs, ordered by vn and by fn; len: 2 * NF
        msg_src_idxes = list()      # list of list; indexes of incoming fn -> vn in msgs, one for each element in msg_trg_idxes; size: (2 * NF, ?), ?=deg(vn) - 1
        embed_trg_idxes = list()    # list; index of each vn -> fn embeddings in x, ordered by vn and by fn; len: 2 * NF
        embed_src_idxes = list()    # list of list; indexes of incoming fn -> vn embeddings in x, one for each element in msg_trg_idxes; size: (2 * NF, ?), ?=deg(vn) - 1
        v2f_scatter_idxes = list()  # list; scatter index for flattened msg_src_idxes & embed_src_idxes; len: ?, ?=sum(deg(vn_i) * (deg(vn_i) - 1)), i=1,...,NV
        trg_scatter_idx = 0

        msg_f2v_per_v_idxes = list()      # list of list; indexes of incoming fn -> vn in msgs, one for each vn; size: (NV, deg(vn))
        f2v_per_v_scatter_idxes = list()  # list; scatter index for flattened msg_f2v_per_v_idxes; len: ?, ?=2 * NF
        f2v_per_v_scatter_idx = 0

        degrees = []
        unary_var_cnt = 0
        for vn in all_vars:
            msg_vn2f_idxes = []
            msg_f2vn_idxes = []

            # rv -> f, cv -> f, f -> rv, f -> cv
            for fn_idx in adj_func_list[vn]:
                _, row, col = all_matrix[fn_idx]
                if row == vn:
                    msg_vn2f_idxes.append(fn_idx)           # rv -> f, offset=0
                    msg_f2vn_idxes.append(fn_idx + 2 * NF)  # f -> rv, offset=2 * NF, skip rv -> f & cv -> f
                else:
                    assert col == vn
                    msg_vn2f_idxes.append(fn_idx + NF)      # cv -> f, offset=NF, skip rv -> f
                    msg_f2vn_idxes.append(fn_idx + 3 * NF)  # f -> cv, offset=3 * NF, skip rv -> f, cv -> f & f -> rv
            msg_f2v_per_v_idxes.append(msg_f2vn_idxes)
            f2v_per_v_scatter_idxes.append([f2v_per_v_scatter_idx] * len(msg_f2vn_idxes))
            f2v_per_v_scatter_idx += 1

            func_list = adj_func_list[vn]
            if len(func_list) == 1:
                unary_var_cnt += 1
                continue
            assert len(func_list) > 1
            degree = len(func_list)
            degrees.append(degree)
            for i in range(degree):
                msg_trg_idxes.append(msg_vn2f_idxes[i])  # target message index in msgs
                tmp = list(msg_f2vn_idxes)               # source message indexes in msgs
                tmp.pop(i)                               # exclude the target fn -> vn
                msg_src_idxes.append(tmp)

                fn_idx = func_list[i]                                               # we use the function node embedding as the q/k
                embed_trg_idxes.append(fn_idx + NV)                                 # index in x, so shift by NV
                embed_src_idxes.append([j + NV for j in func_list if j != fn_idx])  # exclude the target fn
                assert len(embed_src_idxes[-1]) == len(msg_src_idxes[-1]) == degree - 1

                v2f_scatter_idxes.append([trg_scatter_idx] * len(embed_src_idxes[-1]))
                trg_scatter_idx += 1

        assert len(msg_trg_idxes) == len(embed_trg_idxes) == 2 * NF - unary_var_cnt

        msg_src_idxes = [j for i in msg_src_idxes for j in i]
        embed_src_idxes = [j for i in embed_src_idxes for j in i]
        v2f_scatter_idxes = [j for i in v2f_scatter_idxes for j in i]
        msg_f2v_per_v_idxes = [j for i in msg_f2v_per_v_idxes for j in i]
        f2v_per_v_scatter_idxes = [j for i in f2v_per_v_scatter_idxes for j in i]

        assert v2f_scatter_idxes[-1] + 1 == sum(degrees) == 2 * NF - unary_var_cnt
        assert len(msg_src_idxes) == len(embed_src_idxes) == len(v2f_scatter_idxes) == sum([i * (i - 1) for i in degrees])
        assert f2v_per_v_scatter_idxes[-1] + 1 == NV
        assert len(msg_f2v_per_v_idxes) == len(f2v_per_v_scatter_idxes) == 2 * NF

        f_batch = [1] * NF

        cost_tensors = []
        for fn_idx, fn in enumerate(all_matrix):
            fn_tensor = torch.Tensor(fn[0])
            fn_tensor_max = torch.max(fn_tensor) + 1 / scale
            tmp = torch.ones(self.max_dom_size, self.max_dom_size) * fn_tensor_max
            i, j = fn_tensor.shape[0], fn_tensor.shape[1]
            tmp[0:i, 0:j] = fn_tensor
            cost_tensors.append(tmp.unsqueeze(0))
        cost_tensors = torch.cat(cost_tensors, dim=0)

        graph = Data(x=x,
                     edge_index=edge_index,
                     var_embed=var_embed,
                     func_embed=func_embed,
                     msg_hidden=msg_hidden,
                     msgs=msgs,
                     msg_rv2f_idxes=msg_rv2f_idxes,
                     msg_cv2f_idxes=msg_cv2f_idxes,
                     msg_f2rv_idxes=msg_f2rv_idxes,
                     msg_f2cv_idxes=msg_f2cv_idxes,
                     msg_v2f_idxes=msg_v2f_idxes,
                     msg_f2v_idxes=msg_f2v_idxes,
                     msg_trg_idxes=msg_trg_idxes,
                     msg_src_idxes=msg_src_idxes,
                     embed_trg_idxes=embed_trg_idxes,
                     embed_src_idxes=embed_src_idxes,
                     v2f_scatter_idxes=v2f_scatter_idxes,
                     msg_f2v_per_v_idxes=msg_f2v_per_v_idxes,
                     f2v_per_v_scatter_idxes=f2v_per_v_scatter_idxes,
                     f_batch=f_batch,
                     cost_tensors=cost_tensors,
                     sizes=[NV, NF],
                     rv_idxes=rv_idxes,
                     cv_idxes=cv_idxes,
                     func_cnt=int(NF / 2),
                     name=os.path.basename(pth)
                     )
        return graph
