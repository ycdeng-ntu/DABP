import xml.etree.ElementTree as et
from xml.dom import minidom
import os, random

MODE_BINARY = 'BINARY'


def proportional_selection(prob):
    rnd = random.random()
    total = 0
    idx = 0
    while total < rnd:
        total += prob[idx]
        idx += 1
    return idx - 1


class Problem:

    def __init__(self):
        self.agents = dict()  # {agent_id: description}
        self.domains = dict()  # {domain_size: domain_id}
        self.agent_domain_mapping = dict()  # {agent_id: domain_size}
        self.constraints = dict()  # {scope: idx}
        self.functions = dict()  # {idx: obj}
        self.domain_idx = 1
        self.constraint_idx = 1

    def reset(self):
        self.agents = dict()  # {agent_id: description}
        self.domains = dict()  # {domain_size: domain_id}
        self.agent_domain_mapping = dict()  # {agent_id: domain_size}
        self.constraints = dict()  # {scope: idx}
        self.functions = dict()  # {idx: obj}
        self.domain_idx = 1
        self.constraint_idx = 1

    def from_networkx(self, g, domain_size, min_cost=0, max_cost=100, gc=False,
                          weighted=False, decimal=-1):
        self.reset()
        for i in g.nodes:
            self.add_agent(i + 1, domain_size)
        for agnt1, agnt2 in g.edges:
            self.add_constraint([agnt1 + 1, agnt2 + 1], Problem._random_matrix(domain_size, domain_size,
                                                                           min_cost, max_cost, gc, weighted, decimal))

    def random_sensor_net(self, grid_size, domain_size, min_cost=0, max_cost=100, gc=False,
                          weighted=False, decimal=-1):
        self.reset()
        locations = dict()
        agent_id = 1
        for row in range(grid_size):
            for col in range(grid_size):
                self.add_agent(agent_id, domain_size)
                locations[(row, col)] = agent_id
                agent_id += 1
        for row in range(grid_size):
            for col in range(grid_size):
                right = (row, col + 1)
                down = (row + 1, col)
                this_id = locations[(row, col)]
                if right in locations:
                    self.add_constraint([this_id, locations[right]], Problem._random_matrix(domain_size, domain_size,
                                                                           min_cost, max_cost, gc, weighted, decimal))
                if down in locations:
                    self.add_constraint([this_id, locations[down]], Problem._random_matrix(domain_size, domain_size,
                                                                           min_cost, max_cost, gc, weighted, decimal))

    def random_binary(self, nb_agent, domain_size, p1, min_cost=0, max_cost=100, gc=False,
                      weighted=False, decimal=-1):
        assert isinstance(nb_agent, int) and nb_agent > 0
        assert 0 < p1 <= 1
        assert 0 <= min_cost < max_cost
        self.reset()
        for i in range(nb_agent):
            self.add_agent(i + 1, domain_size)
        connected = set()
        remaining = set([x for x in self.agents])
        while len(remaining) > 0:
            agnt1 = random.sample(remaining, 1)[0]
            if len(connected) > 0:
                agnt2 = random.sample(connected, 1)[0]
                self.add_constraint([agnt1, agnt2], Problem._random_matrix(domain_size, domain_size,
                                                                           min_cost, max_cost, gc, weighted, decimal))
            remaining.discard(agnt1)
            connected.add(agnt1)
        edge_cnt = int(nb_agent * (nb_agent - 1) / 2 * p1) - (nb_agent - 1)
        remaining = set([x for x in self.agents])
        while edge_cnt > 0:
            matrix = Problem._random_matrix(domain_size, domain_size, min_cost, max_cost, gc, weighted, decimal)
            while True:
                agents = random.sample(remaining, 2)
                if self.add_constraint(agents, matrix):
                    break
            edge_cnt -= 1

    def random_scale_free(self, nb_agent, domain_size, m1, m2, min_cost=0, max_cost=100, gc=False,
                          weighted=False, decimal=-1):
        assert isinstance(nb_agent, int) and nb_agent > 0
        assert isinstance(m1, int) and isinstance(m2, int)
        assert m2 <= m1 <= nb_agent
        self.reset()
        for i in range(nb_agent):
            self.add_agent(i + 1, domain_size)
        remaining = set(self.agents.keys())
        connected = dict()
        for _ in range(m1):
            agnt1 = random.sample(remaining, 1)[0]
            if len(connected) > 0:
                agnt2 = random.sample(set(connected.keys()), 1)[0]
                self.add_constraint([agnt1, agnt2], Problem._random_matrix(domain_size, domain_size,
                                                                           min_cost, max_cost, gc, weighted, decimal))
                connected[agnt2] += 1
            remaining.discard(agnt1)
            connected[agnt1] = 0 if len(connected) == 0 else 1
        while len(remaining) > 0:
            agnt1 = random.sample(remaining, 1)[0]
            c_agents = [[x, y] for x, y in connected.items()]
            cnt = [x[-1] for x in c_agents]
            for _ in range(m2):
                p_sum = sum(cnt)
                prob = [x / p_sum for x in cnt]
                idx = proportional_selection(prob)
                agnt2 = c_agents[idx][0]
                cnt[idx] = 0
                ok = self.add_constraint([agnt1, agnt2], Problem._random_matrix(domain_size, domain_size,
                                                                           min_cost, max_cost, gc, weighted, decimal))
                assert ok
                connected[agnt2] += 1
            remaining.discard(agnt1)
            connected[agnt1] = m2

    @classmethod
    def _random_matrix(cls, domain1, domain2, min_cost, max_cost, gc=False, weighted=False, decimal=-1, break_tie_ub=.1):
        data = []
        for i in range(domain1):
            data.append([0] * domain2)
            for j in range(domain2):
                if not gc:
                    if decimal <= 0:
                        cost = random.randint(min_cost, max_cost)
                    else:
                        cost = random.random() * (max_cost - min_cost) + min_cost
                        cost = round(cost, decimal)
                elif not weighted:
                    cost = 0 if i != j else 1
                else:
                    cost = 0 if i != j else random.randint(min_cost, max_cost)
                if cost == 0 and gc and decimal > 0:
                    cost = round(random.random() * break_tie_ub, decimal)
                data[i][j] = cost
        return data

    def add_agent(self, agent_id, domain_size, description=None):
        assert isinstance(agent_id, int) and agent_id not in self.agents
        assert isinstance(domain_size, int) and domain_size > 0
        if domain_size not in self.domains:
            self.domains[domain_size] = 'D{}'.format(self.domain_idx)
            self.domain_idx += 1
        self.agents[agent_id] = description if description else 'Agent {}'.format(agent_id)
        self.agent_domain_mapping[agent_id] = domain_size

    def add_constraint(self, scope, function):
        scope = tuple(sorted(scope))
        if scope in self.constraints:
            return False
        self.constraints[scope] = 'C{}'.format(self.constraint_idx)
        self.functions['R{}'.format(self.constraint_idx)] = function
        self.constraint_idx += 1
        return True

    def save(self, path, mode=MODE_BINARY, meta_data=None):
        assert mode in [MODE_BINARY]
        if not meta_data:
            meta_data = dict()
        else:
            assert isinstance(meta_data, dict)
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if os.path.exists(path):
            os.remove(path)
        meta_data['type'] = mode
        root = et.Element('instance')
        et.SubElement(root, 'presentation', meta_data)
        agents = et.SubElement(root, 'agents', {'nbAgents': str(len(self.agents))})
        for agent_id in self.agents:
            et.SubElement(agents, 'agent', {'name': 'A{}'.format(agent_id), 'id': str(agent_id),
                                            'description': self.agents[agent_id]})
        domains = et.SubElement(root, 'domains', {'nbDomains': str(len(self.domains))})
        for domain_size in self.domains:
            et.SubElement(domains, 'domain', {'name': self.domains[domain_size], 'nbValues': str(domain_size)})
        variables = et.SubElement(root, 'variables', {'nbVariables': str(len(self.agents))})
        for agent_id in self.agents:
            et.SubElement(variables, 'variable', {'agent': 'A{}'.format(agent_id), 'name': 'X{}.1'.format(agent_id),
                                                  'domain': self.domains[self.agent_domain_mapping[agent_id]],
                                                  'description': 'Variable X{}.1'.format(agent_id)})

        constraints = et.SubElement(root, 'constraints', {'nbConstraints': str(len(self.constraints))})
        for scp in self.constraints:
            scope_variables = ['X{}.1'.format(x) for x in scp]
            scope = ' '.join(scope_variables)
            arity = str(len(scp))
            et.SubElement(constraints, 'constraint', {'name': self.constraints[scp], 'arity': arity, 'scope': scope,
                                                      'reference': self.constraints[scp].replace('C', 'R')})

        relations = et.SubElement(root, 'relations', {'nbRelations': str(len(self.constraints))})
        for name in self.functions:
            e = et.SubElement(relations, 'relation', {'name': name})
            matrix = self.functions[name]
            parts = []
            for row in range(len(matrix)):
                for col in range(len(matrix[0])):
                    cst = matrix[row][col]
                    if isinstance(cst, int):
                        cst = str(cst)
                    else:
                        cst = f'{cst:.8f}'
                    txt = '{}:{} {}'.format(cst, row + 1, col + 1)
                    parts.append(txt)
            e.text = '|'.join(parts)
        xmlstr = minidom.parseString(et.tostring(root)).toprettyxml(indent="   ")
        with open(path, "w") as f:
            f.write(xmlstr)