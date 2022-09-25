import os

from core.problem import Problem
import argparse
from networkx import newman_watts_strogatz_graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate small-world networks')
    parser.add_argument('-nv', '--num_var', type=int, required=True, help='number of variables')
    parser.add_argument('-d', '--domain_size', type=int, required=True, help='domain size')
    parser.add_argument('-p', '--p', type=float, required=True, help='p')
    parser.add_argument('-k', '--k', type=int, required=True, help='k')
    parser.add_argument('-o', '--output', type=str, required=True, help='output dir')
    parser.add_argument('-n', '--num_instances', type=int, required=False, default=1, help='number of instance to generate')
    args = parser.parse_args()
    for i in range(args.num_instances):
        p = Problem()
        g = newman_watts_strogatz_graph(args.num_var, args.k, args.p)
        p.from_networkx(g=g, domain_size=args.domain_size)
        p.save(os.path.join(args.output, f'{i}.xml'))