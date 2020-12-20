import numpy
import GroupCBUCOVID.NewVersion.covid_farz as utils
import snap
import scipy.sparse
import networkx
import matplotlib.pyplot as pyplot


def generate_demographic_contact_network(N, demographic_data, distancing_scales=[]):
    """
        We generate a FARZ network according to best-guess BU demographic data.
        This represents a 'social distanced' community.
        @param N: number of individuals
        @param demographic_data: dictionary containing the household size distribution and other data
        @param distancing_scales: a list of floats corresponding to desired distancing scales
        @min_num_edges: minimum number of edges for each node
    """
    # Dictionary to store the various generated graphs
    graphs = {}
    household_size_distn    = demographic_data['household_size_distn']

    # Calculate the distribution of household sizes given that the household has more than 1 member:
    household_size_distn_givenGT1 = {key: value/(1-household_size_distn[1]) for key, value in household_size_distn.items()}
    household_size_distn_givenGT1[1] = 0

    # We model two types of students, graduate and undergraduate
    student_types = ['Graduate', 'Undergraduate']
    # 80% of single occupancy households are graduate students, 20% undergraduates
    student_probs_single_occupancy = {'Graduate': 0.8, 'Undergraduate': 0.2}
    # 100% of multi occupancy households are undergraduate students
    student_probs_multi_occupancy = {'Graduate': 0.0, 'Undergraduate': 1.0}

    # 10% of students live in a single, 90% of students live with others
    householdSituations_prob = {}
    householdSituations_prob['SingleOccupancy'] = 0.1
    householdSituations_prob['MultiOccupancy'] = 0.9
    assert(numpy.sum(list(householdSituations_prob.values())) == 1.0), "Household situation probabilities must do not sum to 1"

    # Randomly construct households following the distribution
    # List of dictionaries storing household data/metadata
    households     = []
    # Number of individuals to place in households
    homelessNodes  = N
    curMemberIndex = 0
    while(homelessNodes > 0):

        household = {}
        household['situation'] = numpy.random.choice(list(householdSituations_prob.keys()), p=list(householdSituations_prob.values()))
        household['studentTypes'] = []

        if(household['situation'] == 'SingleOccupancy'):
            # Single occupant
            household['size'] = 1
            # Add one member to the household
            household['studentTypes'].append( numpy.random.choice(list(student_probs_single_occupancy.keys()), p=list(student_probs_single_occupancy.values())) )

        elif(household['situation'] == 'MultiOccupancy'):
            # There's at least two members, select a household size according to the predefined distribution
            household['size'] = min(homelessNodes, max(2, numpy.random.choice(list(household_size_distn_givenGT1), p=list(household_size_distn_givenGT1.values()))) )
            # Add members to the household
            for m in range(household['size'] - len(household['studentTypes'])):
                household['studentTypes'].append( numpy.random.choice(list(student_probs_multi_occupancy.keys()), p=list(student_probs_multi_occupancy.values())) )

        if(len(household['studentTypes']) == household['size']):
            homelessNodes -= household['size']
            households.append(household)
        else:
            # Check for errors
            print("Household size does not match number of student types assigned. " + household['situation'])

    numHouseholds = len(households)

    layer_info  = {'mainLayer': {'studentTypes': ['Graduate', 'Undergraduate']}}

    # Count the number of individuals of each student type in the generated households:
    studentTypes_numInPop = {studentType: numpy.sum([len([st for st in household['studentTypes'] if st==studentType]) for household in households])
                            for studentType in student_types}

    adjMatrices = []
    individualAgeGroupLabels = []
    curidx = 0

    for layerGroup, layerInfo in layer_info.items():
        print("Generating graph for "+layerGroup+"...")

        layerInfo['numIndividuals'] = N

        print(layerInfo['numIndividuals'])

        layerInfo['indices'] = range(curidx, curidx+layerInfo['numIndividuals'])

        curidx += layerInfo['numIndividuals']
        print(curidx)

        individualAgeGroupLabels[min(layerInfo['indices']):max(layerInfo['indices'])] = [layerGroup]*layerInfo['numIndividuals']

        # Generating FARZ graph
        print("Generating FARZ graph...")
        layerInfo['graph'] = utils.generate_farz(farz_params={
                                                        'n': layerInfo['numIndividuals'],
                                                        'm': int(4.5/2),              # mean degree / 2
                                                        'k': int(layerInfo['numIndividuals']/50), # number of communities
                                                        'alpha': 2.0,                 # clustering param
                                                        'gamma': -0.6,                # assortativity param
                                                        'beta':  0.6,                 # prob within community edges
                                                        'r':     1,                   # max num communities node can be part of
                                                        'q':     0.5,                 # probability of multi-community membership
                                                        'phi': 1, 'b': 0.0, 'epsilon': 0.0000001,
                                                        'directed': False, 'weighted': False})


        nodeDegrees = snap.TIntPrV()
        snap.GetNodeOutDegV(layerInfo['graph'], nodeDegrees)
        nodeDegrees = [item.GetVal2() for item in nodeDegrees]

        meanDegree  = numpy.mean(nodeDegrees)
        maxDegree   = numpy.max(nodeDegrees)

        print("Mean Degree: " + str(meanDegree))
        print("Max Degree: " + str(maxDegree))

        def adj_list_from_snapgraph(snapGraph):
          N = snapGraph.GetNodes()
          adj_matrix = numpy.zeros((N, N))
          for e in snapGraph.Edges():
            src = e.GetSrcNId()
            dest = e.GetDstNId()
            adj_matrix[src][dest] = 1
          #print(adj_matrix)
          return adj_matrix

        adj_list = adj_list_from_snapgraph(layerInfo['graph'])

        adjMatrices.append(adj_list)

    # Assemble the NetworkX graph
    A_baseline = scipy.sparse.lil_matrix(scipy.sparse.block_diag(adjMatrices))
    G_baseline = networkx.from_scipy_sparse_matrix(A_baseline)
    graphs['baseline'] = G_baseline

    # GENERATE SOCIALLY DISTANCED GRAPHS (in-household connections unaffected)
    # In the full isolation graph, all nodes isolate on their own, graph is empty
    graphs['full_isolation'] = networkx.classes.function.create_empty_copy(G_baseline)
    # Create a copy of the baseline graph
    G_baseline_NODIST   = graphs['baseline'].copy()
    # Randomly remove a number of edges drawn from an exponential distribution
    for dist_scale in distancing_scales:
        graphs['distancingScale'+str(dist_scale)] = custom_exponential_graph(G_baseline_NODIST, scale=dist_scale)




    # Add edges between housemates to strongly connect households
    for layerGroup, layerInfo in layer_info.items():
        layerInfo['selection_indices'] = list(layerInfo['indices'])

    individualStudentTypeLabels = [None]*N

    # Strongly connect the nodes selected for each household by adding edges to the adjacency matrix.
    for household in households:
        household['indices'] = []
        for studentType in household['studentTypes']:
            ageGroupIndices = next(layer_info[item]['selection_indices'] for item in layer_info if studentType in layer_info[item]["studentTypes"])
            memberIndex     = ageGroupIndices.pop()
            household['indices'].append(memberIndex)
            individualStudentTypeLabels[memberIndex] = studentType

        for memberIdx in household['indices']:
            nonselfIndices = [i for i in household['indices'] if memberIdx!=i]
            for housemateIdx in nonselfIndices:
                # Apply to all graphs
                for graphName, graph in graphs.items():
                    graph.add_edge(memberIdx, housemateIdx)

    return graphs, individualStudentTypeLabels, households

def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    """
        We modify the graph by probabilistically dropping some edges from each node.
        This represents a 'social distanced' community.
        @param base_graph: a SNAP/NetworkX type random graph
        @param scale: number of nodes in the graph
        @min_num_edges: minimum number of edges for each node
    """
    if type(base_graph) == networkx.classes.graph.Graph:
        graph = base_graph.copy()
        for node in graph:
            neighbors = list(graph[node].keys())
            if(len(neighbors) > 0):
                quarantineEdgeNum = int(max(min(numpy.random.exponential(scale=scale, size=1), len(neighbors)), min_num_edges) )
                quarantineKeepNeighbors = numpy.random.choice(neighbors, size=quarantineEdgeNum, replace=False)
                for neighbor in neighbors:
                    if(neighbor not in quarantineKeepNeighbors):
                        graph.remove_edge(node, neighbor)
        return graph
    # If graph is SNAP graph
    elif type(base_graph) == snap.PUNGraph:
        for NI in base_graph.Nodes():
            list_neighbors = []
            for Id in NI.GetOutEdges():
                list_neighbors.append(Id)
            if(len(list_neighbors) > 0):
                quarantineEdgeNum = int(max(min(numpy.random.exponential(scale=scale, size=1), len(list_neighbors)), min_num_edges))
                quarantineKeepNeighbors = numpy.random.choice(list_neighbors, size=quarantineEdgeNum, replace=False)
                temp = []
                DstNId_temp =[]
                for Id in NI.GetOutEdges():
                    DstNId_temp.append(Id)
                    temp.append(NI.GetId())
                for i in range(len(temp)-1):
                    if(DstNId_temp[i] not in quarantineKeepNeighbors):
                        base_graph.DelEdge(DstNId_temp[i], temp[i])
        return base_graph
    # Report error to user and return the base graph
    else:
        print("Input must be a SNAP graph or a NetworkX graph")
        return base_graph
