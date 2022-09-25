import alg.constant as constant
from alg.data import COP_Dataset

from torch_geometric.loader import DataLoader as torch_DataLoader
import argparse

from alg.run import DABP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DABP')
    parser.add_argument('-D', '--dir', type=str, required=True, help='problem directory')
    parser.add_argument('-d', '--domain_size', type=int, required=True, help='maximum domain size')
    parser.add_argument('-nh', '--num_head', type=int, required=False, default=4, help='number of attention heads')
    parser.add_argument('-mt', '--max_it', type=int, required=False, default=1000, help='maximum iteration limit')
    parser.add_argument('-ut', '--upd_it', type=int, required=False, default=20, help='update interval')
    parser.add_argument('-et', '--eff_it', type=int, required=False, default=2, help='number of effective iterations')
    parser.add_argument('-rs', '--restart', type=int, required=False, default=20, help='number of restarts')
    parser.add_argument('-gid', '--gpu_id', type=int, required=False, default=0, help='GPU ID')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed log')
    args = parser.parse_args()
    constant.gpu_id = args.gpu_id

    print(f'Working dir: {args.dir}')
    param = {'batch_size': 1, 'shuffle': False}
    dataset = COP_Dataset(args.dir, max_dom_size=args.domain_size)
    dataloader = torch_DataLoader(dataset=dataset, **param)
    dabp = DABP(dom_size=args.domain_size,
                num_head=args.num_head,
                max_iterations=args.max_it,
                update_interval=args.upd_it,
                eff_iterations=args.eff_it,
                num_restart=args.restart)

    for prob in dataloader:
        cost = dabp.solve(prob, verbose=args.verbose)
        print(f'Solution Cost of Problem {prob.name[0]}: {cost:.2f}')
