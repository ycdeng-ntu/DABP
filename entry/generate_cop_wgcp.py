import os

from core.problem import Problem
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate random COPs and WGCPs')
    parser.add_argument('-nv', '--num_var', type=int, required=True, help='number of variables')
    parser.add_argument('-p1', '--density', type=float, required=True, help='graph density')
    parser.add_argument('-d', '--domain_size', type=int, required=True, help='domain size')
    parser.add_argument('-wgc', '--weighted_graph_coloring', action='store_true', help='generate weighted graph coloring problems')
    parser.add_argument('-o', '--output', type=str, required=True, help='output dir')
    parser.add_argument('-n', '--num_instances', type=int, required=False, default=1, help='number of instance to generate')
    args = parser.parse_args()
    for i in range(args.num_instances):
        p = Problem()
        gc = args.weighted_graph_coloring
        dec = 0 if not gc else 8
        p.random_binary(args.num_var, args.domain_size, args.density, gc=gc, weighted=gc, decimal=dec)
        p.save(os.path.join(args.output, f'{i}.xml'))