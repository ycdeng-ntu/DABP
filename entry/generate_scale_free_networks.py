import os

from core.problem import Problem
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate scale-free networks')
    parser.add_argument('-nv', '--num_var', type=int, required=True, help='number of variables')
    parser.add_argument('-d', '--domain_size', type=int, required=True, help='domain size')
    parser.add_argument('-m0', '--m0', type=int, required=True, help='m0')
    parser.add_argument('-m1', '--m1', type=int, required=True, help='m1')
    parser.add_argument('-o', '--output', type=str, required=True, help='output dir')
    parser.add_argument('-n', '--num_instances', type=int, required=False, default=1, help='number of instance to generate')
    args = parser.parse_args()
    for i in range(args.num_instances):
        p = Problem()
        p.random_scale_free(args.num_var, args.domain_size, args.m0, args.m1)
        p.save(os.path.join(args.output, f'{i}.xml'))