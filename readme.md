# Dependencies
- **PyTorch 1.9.0**
- **PyTorch Geometric 2.0.2**
- **Networkx 2.5**

# Directory structure
- `alg` contains the implementation of DABP
- `core` contains the core functionalities (i.e., problem generation & parsing) to run the simulation
- `entry` contains the entry points

# How to run the code

See the command line interface of `*.py` in `entry`.

Example:

We first generate 100 random COPs and store them in `./rndCOPs` by:

`python -um entry.generate_cop_wgcp -nv 60 -d 15 -p1 0.25 -o ./rndCOPs -n 100`

Then we solve each of them via DABP by:

`python -um entry.run_dabp -D ./rndCOPs -d 15 -v`