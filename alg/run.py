from time import perf_counter
from alg.data import *
from alg.model import AttentiveBP
from torch.optim import AdamW

from alg.constant import device

from statistics import mean, stdev

class DABP:
    def __init__(self, dom_size, num_head=4, max_iterations=1000, update_interval=20, eff_iterations=2, num_restart=20):
        self.abp = AttentiveBP(in_channels=12, out_channels=16, num_heads=num_head, msg_dim=dom_size)
        self.abp.double()
        self.num_head = num_head
        self.dom_size = dom_size
        self.max_iterations = max_iterations
        self.update_interval = update_interval
        self.eff_iterations = eff_iterations
        self.num_restart = num_restart

    def solve(self, cop_data, verbose=True):
        abp = AttentiveBP(in_channels=12, out_channels=16, num_heads=self.num_head, msg_dim=self.dom_size)
        abp.double()
        abp.load_state_dict(self.abp.state_dict())
        abp.to(device)
        optimizer = AdamW(abp.parameters(), lr=1e-4, weight_decay=5e-5)
        best_cost = 99999999
        start = perf_counter()
        for rs in range(self.num_restart):
            abp.preprocess(cop_data)
            c_best_cost = 99999999
            for phase in range(int(self.max_iterations / self.update_interval)):
                optimizer.zero_grad()
                losses, costs = abp(n_iter=self.update_interval)

                idxes = sorted(range(len(losses)), key=costs.__getitem__)
                best_cost = min([best_cost] + costs)
                c_best_cost = min([c_best_cost] + costs)
                top_k_loss = 0
                for i in range(self.eff_iterations):
                    idx = idxes[i]
                    top_k_loss = top_k_loss + losses[idx]
                top_k_loss = top_k_loss / self.eff_iterations
                top_k_loss.backward()
                optimizer.step()
                costs = [int(round(x * scale)) for x in costs]
                cov = stdev(costs) / mean(costs)
                if verbose:
                    print(f'Update #{phase + 1} of Restart #{rs + 1}: Loss: {top_k_loss.item():.4f}|\tCurrent Best Cost:{c_best_cost * scale / cop_data.func_cnt.item():.2f}|\tBest Cost:{best_cost * scale / cop_data.func_cnt.item():.2f}|\tElapse:{perf_counter() - start:.2f}')
                if cov < 0.001:
                    break
        return best_cost * scale / cop_data.func_cnt.item()