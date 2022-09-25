import xml.etree.ElementTree as ET


def parse(path, scale=1):
    agents = {}
    root = ET.parse(path).getroot()
    ele_agents = root.find('agents')
    for ele_agent in ele_agents.findall('agent'):
        id = ele_agent.get('name')
        agents[id] = [id, 0]
    domains = {}
    ele_domains = root.find('domains')
    for ele_domain in ele_domains.findall('domain'):
        id = ele_domain.get('name')
        nb_values = ele_domain.get('nbValues')
        domains[id] = int(nb_values)

    ele_variables = root.find('variables')
    for ele_variable in ele_variables.findall('variable'):
        agent_id = ele_variable.get('agent')
        domain_id = ele_variable.get('domain')
        agents[agent_id][-1] = domains[domain_id]
    constraints = {}
    relations = {}
    ele_constraints = root.find('constraints')
    for ele_constraint in ele_constraints.findall('constraint'):
        id = ele_constraint.get('name')
        scope = ele_constraint.get('scope').split(' ')
        scope = ['A' + s[1: -2] for s in scope]
        reference = ele_constraint.get('reference')
        constraints[id] = scope
        relations[reference] = id

    ele_relations = root.find('relations')
    all_matrix = []
    for ele_relation in ele_relations.findall('relation'):
        id = ele_relation.get('name')
        content = ele_relation.text.split('|')
        first_constraint = []
        for tpl in content:
            cost, values = tpl.split(':')
            cost = float(cost) / scale
            values = [int(s) for s in values.split(' ')]
            while len(first_constraint) < values[0]:
                first_constraint.append([])
            row = first_constraint[values[0] - 1]
            while len(row) < values[1]:
                row.append(0)
            row[values[1] - 1] = cost
        pair = constraints[relations[id]]
        all_matrix.append((first_constraint, pair[0], pair[1]))
    all_vars = []
    for data in agents.values():
        all_vars.append(tuple(data))
    return all_vars, all_matrix
