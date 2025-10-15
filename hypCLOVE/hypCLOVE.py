__version__ = '1.0.1'

from collections import defaultdict
from itertools import combinations

import warnings
import random as rand
import math
import numpy as np
import networkx as nx
from scipy.special import zeta

from community import best_partition
from community import induced_graph as induced_graph
from community import generate_dendrogram
from community import partition_at_level


import igraph as ig
import leidenalg as la


# Calculate Complementary Cumulative Distribution Function (CCDF)
def calculate_ccdf(degree_list):
    hist, bins = np.histogram(degree_list, bins=range(int(np.min(degree_list)), int(np.max(degree_list)) + 2), density=True)
    return bins[:-1], np.cumsum(hist[::-1])[::-1]  # degree_axis and CCDF

def hill_estimator(degree_list, degree_axis, ccdf, min_samples=100):
    sorted_degrees = sorted(set(degree_list))
    gamma_vals, d_vals = [], []
    
    num_of_diff_degs = len(sorted_degrees)
    
    if num_of_diff_degs < min_samples:
        min_samples = num_of_diff_degs
    
    for kmin in sorted_degrees:
        if kmin <= 0.5:  # Ensure kmin is valid
            continue
        
        filtered_degrees = degree_list[degree_list >= kmin]
        if len(filtered_degrees) < min_samples: 
            continue
        
        # Handle small or zero values inside log
        gamma = 1 + len(filtered_degrees) / np.sum(np.log(np.maximum(filtered_degrees / (kmin - 0.5), 1e-10)))
        
        kmin_idx = np.where(degree_axis == kmin)[0][0]
        tail_degrees = degree_axis[kmin_idx:]
        
        # Handle small or zero ccdf and tail_degrees
        valid_ccdf = np.maximum(ccdf[kmin_idx:], 1e-10)
        valid_tail_degrees = np.maximum(tail_degrees, 1e-10)
        
        constant = np.exp(np.mean(np.log(valid_ccdf) + (gamma - 1) * np.log(valid_tail_degrees))) * (gamma - 1)

        fitted_ccdf = (1 / ((gamma - 1) * zeta(gamma, kmin))) * (valid_tail_degrees ** (1 - gamma))
        actual_ccdf = valid_ccdf / (zeta(gamma, kmin) * constant)

        gamma_vals.append(gamma)
        d_vals.append(np.max(np.abs(actual_ccdf - fitted_ccdf)))
    
    return gamma_vals[np.argmin(d_vals)]

# Fit degree decay exponent
def fit_degree_decay_exponent(G, min_samples):
    if min_samples is None:
        min_samples = 100
    degree_list = np.array([deg for _, deg in G.degree], dtype=int)
    degree_axis, ccdf = calculate_ccdf(degree_list)
    return hill_estimator(degree_list, degree_axis, ccdf, min_samples)
    
# Converting community structure from a dict format (partitioning -> part) to the the format of lists of lists (communities -> comm)
def convert_part_to_comm(partitioning):
    """
    Converts a dictionary mapping nodes to communities into a list of communities,
    where each community is represented as a list of its member nodes. 
    The output communities are sorted by community identifiers.
    
    Parameters:
        partitioning (dict): A dictionary where keys are node identifiers and values are community identifiers.

    Returns:
        list: A list of communities, each represented as a list of nodes.
    """
    # Grouping keys by their values
    communities = defaultdict(list)
    for node, community in partitioning.items():
        communities[community].append(node)

    # Return communities sorted by their keys
    return [sorted(communities[comm]) for comm in sorted(communities)]

# Louvain algorithm for community detection
def Louvain(G, seed = None):
    return best_partition(G, random_state = seed)

# Leiden algorithm for community detection (using the igraph package)
def Leiden(G, seed = None):
    """
    Applies the Leiden community detection algorithm to a given graph.

    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        dict: A dictionary mapping each node to its corresponding community.
    """
    # Convert the NetworkX graph to an igraph object
    g = ig.Graph.from_networkx(G)
    # Find the partition using the Leiden algorithm
    partition = la.find_partition(g, la.ModularityVertexPartition, seed = seed)
    
    # Return a dictionary mapping nodes to their community memberships
    return {node: community for node, community in zip(G.nodes(), partition.membership)}

# Connected component partitioning: partioning the network according to its connected components
def connected_components_partitioning(G):
    """Partition the graph G into connected components and map each node to its component index."""
    # Create a dictionary mapping each node to its component index
    return {node: cc_index for cc_index, component in enumerate(nx.connected_components(G)) for node in component}

# defining the sizes of angular sectors corresponding to the communities
def update_angular_coords(G, community_order, communities, angs, comm_sector_size_prop='node_num'):
    """
    Update the angular coordinates of nodes in a graph based on their community memberships.
    
    Parameters:
    G: NetworkX graph
        The input graph where nodes are assigned to communities.
    community_order: list
        The order of communities for which angular coordinates are updated.
    communities: list of lists
        A list where each sublist contains nodes belonging to the same community.
    angs: list
        A list containing the current angular coordinates of the (super-)nodes corresponding to the communities.
    comm_sector_size_prop: str
        Determines the method of calculating the sector size; 
        'node_num' uses the number of nodes, i.e. the associated sector size is prop. to the num of nodes inside 
        the corresponding community. 
        'edge_num' uses the number of edges in a similar fashion.
        
    Returns:
    list
        Updated list of angular coordinates for the (super-)nodes corresponding to the communities.
    """
    # Get the number of nodes and edges in the graph
    N = len(G.nodes())
    E = len(G.edges())
    
    # Initialize the previous sector boundary using the angular coordinate of the first community's first node
    prev_sector_bound = angs[communities[community_order[0]][0]]

    # Iterate through the ordered list of communities
    for commID in community_order:
        # Get the list of nodes in the current community
        commnodes = communities[commID]
        commlen = len(commnodes)  # Number of nodes in the current community

        # Calculate the angular sector size based on the specified property
        if comm_sector_size_prop == 'node_num':
            # Size is proportional to the number of nodes in the community
            angular_sector_size = 2. * math.pi * commlen / N
        else:  # 'edge_num'
            # Size is proportional to the total degree of nodes in the community
            k = sum(G.degree(node) for node in commnodes)  # Total degree of nodes in the community
            angular_sector_size = 2. * math.pi * k / (2. * E)  # Proportional to total edges

        # Update the angular coordinates for each node in the community
        for node in commnodes:
            angs[node] = prev_sector_bound

        # Update the previous sector boundary for the next community
        prev_sector_bound += angular_sector_size
    
    return angs  # Return the updated angular coordinates

# EXPONENTIAL COARSENING (default for coarsening)
def exponential_coarsening(ind_G):
    """
    Coarsens the input graph by creating a new graph where edges are weighted 
    according to an exponential function based on the weights of the original graph's edges.
    
    If community arrangement is solved by TSP then any function can be used that depends on ind_G and the obtained weights satisfies 
    triangle inequality
    
    Parameters:
    ind_G: NetworkX induced supergraph
        The input supergraph is a weighted graph, where nodes are communities detected earlier and 
        the edges between nodes are weighted based on the respective number of intra- and intercommunity links.

    Returns:
    super_G: NetworkX graph
        A new supergraph where the nodes are the same as the original, but the edges are reweighted 
        based on the exponential function of their weights.
    """
    
    # Calculate the total weight of all edges in the input graph
    E = sum([w for c1, c2, w in ind_G.edges(data='weight')])

    # Create a new graph for the coarsened representation
    super_G = nx.Graph()
    
    # Get lists of nodes and edges from the original graph
    supernodes = list(ind_G.nodes())
    superedges = list(ind_G.edges())
    
    # Iterate through all pairs of nodes to create edges in the new graph
    for (node1, node2) in combinations(supernodes, 2):
        # Check if there is an edge between node1 and node2 in the original graph
        if (node1, node2) in superedges or (node2, node1) in superedges:
            # Get the weights of self-loops for the nodes (default to 0 if not present, although not necessary)
            ind_Gw1 = ind_G.get_edge_data(node1, node1, default={'weight': 0.})['weight']
            ind_Gw2 = ind_G.get_edge_data(node2, node2, default={'weight': 0.})['weight']
            
            # If either node has no self-loop weight (intra community links), add an edge with a default weight of 1
            if ind_Gw1 == 0. or ind_Gw2 == 0.:
                super_G.add_edge(node1, node2, weight=1.)
            else:
                # Calculate the new weight using the exponential function based on original weights
                new_weight = math.exp(-2. * E * ind_G[node1][node2]['weight'] / (ind_Gw1 * ind_Gw2)) + 1.
                super_G.add_edge(node1, node2, weight=new_weight)
        else:
            # If no edge exists between node1 and node2, assign a default weight of 2
            super_G.add_edge(node1, node2, weight=2.)        
    # TRINGLE INEQUALITY IS GUARANTEED -> METRIC TSP CAN BE USED LATER ON!
    return super_G

def update_community_order(prev_community_order, prev_commID, sub_community_order):
    """
    Updates the order of communities by replacing a specified community's position 
    with the order of its sub-communities.

    Parameters:
    prev_community_order: list
        The order of the communities at a higher level.
    prev_commID: int
        The ID of a given community at a higher level.
    sub_community_order: list
        The order of the sub-communities within the community identified by prev_commID.

    Returns:
    new_order: list
        The updated order of communities with sub-communities in place of the specified community.
    """
    
    # Create a new order list based on the previous community order
    new_order = list(prev_community_order)
    
    # Find the position of the community ID to be replaced (negated value)
    pos = new_order.index(-1 * prev_commID)
    
    # Replace the identified community with its sub-community order
    return new_order[0:pos] + sub_community_order + new_order[pos + 1:]



def sub_community_chirality(sub_community_order):
    """
    Determines the chirality of sub-communities by organizing them 
    into a clockwise and anti-clockwise order, excluding virtual nodes.

    Parameters:
    sub_community_order: list
        The order of sub-communities, which may include virtual nodes.

    Returns:
    list
        A combined list of sub-communities arranged in clockwise and anti-clockwise order.
    """
    # Copy the original sub-community order to avoid modifying it directly
    c_sub_community_order = list(sub_community_order)
    
    # Get the indices of sub-communities sorted in ascending order
    argsorted_commIDs = np.argsort(c_sub_community_order)
    
    # Identify the two largest community IDs for chirality determination
    clockwise_commID = argsorted_commIDs[-1]
    anti_clockwise_commID = argsorted_commIDs[-2]
    
    # Extract the clockwise order of sub-communities
    clockwise_order = c_sub_community_order[clockwise_commID:anti_clockwise_commID]
    del clockwise_order[0]  # Remove the first element (virtual node)
    
    # Extract the anti-clockwise order of sub-communities
    anti_clockwise_order = c_sub_community_order[anti_clockwise_commID:]
    del anti_clockwise_order[0]  # Remove the first element (virtual node)
    anti_clockwise_order.reverse()  # Reverse the order for proper arrangement
    
    # Return the combined order of clockwise and anti-clockwise sub-communities
    
    return clockwise_order + anti_clockwise_order

def get_neighboring_communities(commID, community_order, anchor_num=1):
    """
    Retrieves the neighboring communities surrounding a specified community ID.

    Parameters:
    commID: int
        The ID of the community for which neighboring communities should be found
    community_order: list
        The current order of communities.
    anchor_num: int, optional
        The number of neighboring communities to return on each side (default is 1).

    Returns:
    tuple
        A tuple containing two lists:
        - Communities to the left (preceding) of commID
        - Communities to the right (following) of commID
    """
    
    orderlen = len(community_order)
    
    # Create an extended list of community_order to facilitate neighbor lookup
    community_order3x = list(community_order) * 3
    
    # Find the position of the specified community ID
    pos = community_order.index(commID)
    
    # Get neighboring communities within the specified anchor_num
    return (
        community_order3x[orderlen + pos - anchor_num: orderlen + pos],  # Left neighbors
        community_order3x[orderlen + pos + 1: orderlen + pos + 1 + anchor_num]  # Right neighbors
    )

def create_sub_net(G, next_partitioning, communities, commID, community_order, anchor_num=1):
    """
    Create a sub-network for a given community with optional virtual nodes representing neighboring communities
    on the upper level.
    
    Parameters:
    G : networkx.Graph
        The original graph.
    next_partitioning : dict
        Partitioning at the previous level (l-1), mapping nodes to community IDs.
    communities : dict
        Communities at the current level (l), mapping community IDs to lists of nodes.
    commID : int
        The community ID for which the sub-network is created.
    community_order : list
        The ordered list of communities at the current level (l).
    anchor_num : int, optional
        The number of neighboring communities to consider when creating virtual nodes (default: 1).
    
    Returns:
    G_sub : networkx.Graph
        Induced subgraph consisting of the community and neighboring communities as virtual nodes.
    subnode2comms : dict
        Mapping of nodes to their community IDs, including virtual communities for neighboring nodes.
    """

    commnodes = communities[commID]
    
    # Return only the induced subgraph if anchor_num is 0
    if anchor_num == 0:
        return nx.induced_subgraph(G, commnodes), {node: next_partitioning[node] for node in commnodes}

    # Get neighboring community nodes (on the upper level) based on the anchor_num parameter
    commIDlefts, commIDrights = get_neighboring_communities(commID, community_order, anchor_num)
    commleftnodes = [node for IDleft in commIDlefts for node in communities[IDleft]]
    commrightnodes = [node for IDright in commIDrights for node in communities[IDright]]

    # Induced subgraph containing the current community and neighboring communities (anchors) on the upper level
    subcommnodes = commleftnodes + commnodes + commrightnodes
    G_sub = nx.induced_subgraph(G, subcommnodes)
    
    max_key = max(next_partitioning.values())
    
    # Assign virtual community IDs to neighboring nodes
    subnode2comms = {node: next_partitioning[node] for node in commnodes}
    subnode2comms.update({node: max_key + 2 for node in commleftnodes})
    subnode2comms.update({node: max_key + 1 for node in commrightnodes})

    return G_sub, subnode2comms

def community_arrange(G, partitioning, coarsening_method=exponential_coarsening, arrangement_method='tsp_christofides', seed = None):
    """
    Arranges community structure in a graph using TSP-based (or a customized) optimization method.
    Parameters:
    G (networkx.Graph): The original graph.
    partitioning (dict): Dictionary mapping nodes to partitions.
    coarsening_method (callable, optional): Function to coarsen the graph. Default is exponential_coarsening.
    arrangement_method (str or callable): If str then community arrangemenet is solved by the TSP. 
        Supports 'tsp_christofides', 'tsp_greedy', 'tsp_simulated_annealing', and 'tsp_threshold_accepting'. 
        Additional boosting methods ('+sa' or '+ta') can be appended.
        The arrangement_method is customizable; it accepts any callable method that returns a valid order of the communities.
    
    Returns:
    list: Optimized path of nodes in the graph according to the selected TSP (or the customized) method.
    """
    
    # Step 1: Create induced graph based on partitioning
    induced_G = induced_graph(partitioning, G)
    if not coarsening_method is None:
        # Step 2: Apply coarsening to get a TSP-compatible graph
        method_compatible_G = coarsening_method(induced_G)
    else:
        # Step 2: No coarsening
        # Create a new graph with the same edges but without weights
        method_compatible_G = nx.Graph()
        method_compatible_G.add_nodes_from(induced_G.nodes)  # Add the nodes
        method_compatible_G.add_edges_from([(u, v, {'weight': 1.}) for u,v in induced_G.edges])  # Add the edges (ignores attributes)
  
    # if arrangement_method is a str the arrangement is solved by the TSP
    if isinstance(arrangement_method, str):    
        # When TSP appplied, method_compatible_G should be a complete graph
        # this is only a necessary, but not sufficient condition for the applicability of TSP
        # sufficient would be tri inequality
        # fast check of completeness of method_compatible_G
        # Ensure anchor_num is non-negative and integer
        if not 2*len(method_compatible_G.edges) == len(method_compatible_G.nodes)*(len(method_compatible_G.nodes)-1):
            raise ValueError(f"'When TSP is used for solving the arrangement problem the coarsened graph should be complete. Got N={len(method_compatible_G.nodes)} nodes and E={len(method_compatible_G.edges)} edges.")
        
        # Step 3: Parse arrangement_method for solution and optional boosting method
        try:
            solve_method, boosting = arrangement_method.split('+')
        except ValueError:
            solve_method, boosting = arrangement_method, None
        
        # Step 4: Solve the TSP using the selected method
        if solve_method == 'tsp_christofides':
            tsp_func = nx.approximation.traveling_salesman_problem
            path = tsp_func(method_compatible_G, weight='weight', cycle=True)
        elif solve_method == 'tsp_greedy':
            path = nx.approximation.greedy_tsp(method_compatible_G, weight='weight')
        elif solve_method == 'tsp_simulated_annealing':
            init_cycle = list(method_compatible_G.nodes) 
            rand.shuffle(init_cycle)
            init_cycle += [init_cycle[0]]
            path = nx.approximation.simulated_annealing_tsp(method_compatible_G, init_cycle=init_cycle, weight='weight', seed = seed)
        elif solve_method == 'tsp_threshold_accepting':
            init_cycle = list(method_compatible_G.nodes) 
            rand.shuffle(init_cycle)
            init_cycle += [init_cycle[0]]
            path = nx.approximation.threshold_accepting_tsp(method_compatible_G, init_cycle=init_cycle, weight='weight', seed = seed)
        else:
            raise ValueError(f"Invalid TSP solving method: {solve_method}")

        # Step 5: Apply boosting if specified
        if boosting == 'sa':
            path = nx.approximation.simulated_annealing_tsp(method_compatible_G, init_cycle=path, weight='weight', seed = seed)
        elif boosting == 'ta':
            path = nx.approximation.threshold_accepting_tsp(method_compatible_G, init_cycle=path, weight='weight', seed = seed)
        elif boosting is not None:
            raise ValueError(f"Invalid boosting method: {boosting}")

        # Step 6: Remove the last element to return an open path instead of a cycle
        path = path[:-1]


    # if arrangement is _not_ based on the TSP. In this case, arragement_method must be a callable variable that returns a valid 
    else:
        path = arrangement_method(G = method_compatible_G, seed = seed)
    
    # Step 7: Adjust the path to ensure sorted order based on partitioning
    # (Rearrange the path for better ordering based on partitioning)
    max_idx = np.argmax(path)
    optimized_path = np.concatenate(
        (np.array(path[max_idx:], dtype=np.int64), np.array(path[:max_idx], dtype=np.int64))
    )
        
    return method_compatible_G, list(optimized_path)

def nodewise_arrangement_ang_distribute_lowest_level(G, lowest_communities, community_order, angs, 
                                                     arrangement_method='tsp_christofides', coarsening_method = exponential_coarsening, 
                                                     comm_sector_size_prop = 'node_num', anchor_num = 1, seed = None):
    """
    Distributes angular coordinates among nodes at the lowest community level, using TSP-based optimization.

    Parameters:
    G (networkx.Graph)
    lowest_communities (list): List of node communities at the lowest level.
    community_order (list): Order of the communities at the level above the lowest.
    angs (dict): Angular coordinates for nodes.
    arrangement_method (str, optional): TSP solving method ('tsp_christofides', 'tsp_greedy', etc.). 
    coarsening_method (callable, optional): Method to coarsen the graph. Default is exponential_coarsening.
    comm_sector_size_prop (str, optional): Proportion strategy for sector sizes ('node_num' or others). Default is 'node_num'.
    anchor_num (int, optional): anchor_num level for creating sub-networks. Default is 1.

    Returns:
    dict: Updated angular coordinates for nodes in the graph.
    """
    
    subcommnum = 0
    fined_tsp_order = [-1*order_com for order_com in community_order]
    new_partition = {}
    
    # Iterate over each community in the community order
    for commID in range(len(community_order)):
        # Step 1: Create subgraph for the current community without virtual nodes
        G_sub_no_virtual = nx.induced_subgraph(G, lowest_communities[commID])
        # Step 2: Create a local subpartition of nodes in the community
        local_subpartition = {node: i for (i,node) in enumerate(lowest_communities[commID])}
        subpartition = {k: v + subcommnum for k,v in local_subpartition.items()}

        new_partition = new_partition | subpartition
        
        # Step 3: Solve TSP on the subgraph
        if len(set(subpartition.values())) > 1:
            if anchor_num == 0:
                # Simple TSP arrangement without virtual nodes since anchor_num = 0
                sub_tsp_G, sub_tsp_order = community_arrange(G_sub_no_virtual, subpartition, coarsening_method = coarsening_method, 
                arrangement_method = arrangement_method, seed = seed)
            else:
                # Create a subnetwork with anchor_num and solve TSP with additional chirality checks
                G_sub, subnode2comms = create_sub_net(G, subpartition, lowest_communities, commID, community_order, 
                                                          anchor_num = anchor_num)
                sub_tsp_G, sub_tsp_order = community_arrange(G_sub, subnode2comms, coarsening_method = coarsening_method, 
                                                                arrangement_method = arrangement_method, seed = seed)

                sub_tsp_order = sub_community_chirality(sub_tsp_order)
        else:
            # If there's only one node, TSP order is trivial
            sub_tsp_order = [list(set(subpartition.values()))[0]]
        
        # Step 4: Update the overall community order with the new sub-community TSP order
        fined_tsp_order = update_community_order(fined_tsp_order, commID, sub_tsp_order)
        subcommnum = max(subpartition.values()) + 1
    
    # Step 5: Prepare final outputs
    tsp_order = list(fined_tsp_order)  # Convert final TSP order list
    partition = new_partition.copy()  # Create a copy of the partition
    communities = convert_part_to_comm(partition).copy()  # Convert the partition to communities
    
    # Step 6: Update angular coordinates based on the TSP order and communities
    angs = update_angular_coords(G, tsp_order, communities, angs, comm_sector_size_prop = comm_sector_size_prop)
    return angs

def rand_ang_distribute_lowest_level(lowest_communities, angs):
    """
    Randomly distributes angular coordinates to nodes in the lowest level communities.

    Parameters:
    lowest_communities (list): List of lowest-level node communities.
    angs (dict): Dictionary of angular coordinates for each node.

    Returns:
    dict: Updated angular coordinates with random distribution.
    """
    N = len(angs) 
    increment = 2.*math.pi/N
    for comm in lowest_communities:
        N_comm = len(comm)
        rand.shuffle(comm)
        for i in range(N_comm):
            angs[comm[i]] += increment*i
    return angs



def degreedy_ang_distribute_lowest_level(G, lowest_communities, community_order, angs, anchor_num = 1):
    """
    Distributes angular coordinates based on a degreedy algorithm for lowest-level communities.

    Parameters:
    G (networkx.Graph): The original graph.
    lowest_communities (list): List of node communities at the lowest level.
    community_order (list): Order of the communities.
    angs (dict): Angular coordinates for nodes.
    anchor_num (int, optional): anchor_num for determining neighboring communities. Default is 1.

    Returns:
    dict: Updated angular coordinates for nodes.
    """
    N = len(G.nodes())
    for upper_commID in community_order:
        commnodelist, commsize = np.array(lowest_communities[upper_commID]), len(lowest_communities[upper_commID])
        commnodedegslist = np.array(G.degree(commnodelist))
        commnodedegorder = commnodelist[np.flip(np.argsort(commnodedegslist,axis=0))[:,0]]
        
        uppercommIDlefts, uppercommIDrights = get_neighboring_communities(upper_commID, community_order, anchor_num = anchor_num)
        commleftnodes, commrightnodes = [], []
        for uIDleft in uppercommIDlefts:
            commleftnodes += list(lowest_communities[uIDleft])
        for uIDright in uppercommIDrights:
            commrightnodes += list(lowest_communities[uIDright])    
        
        subcommunity_order = [commnodedegorder[0]]
        lefts, rights = 0, 0
        for ID in range(1, commsize):
            leftconnections = len([(commnodedegorder[ID], leftnode) for leftnode in commleftnodes if (commnodedegorder[ID], leftnode) in G.edges()])
            rightconnections = len([(commnodedegorder[ID], rightnode) for rightnode in commrightnodes if (commnodedegorder[ID], rightnode) in G.edges()]) 
            
            if leftconnections > rightconnections:
                subcommunity_order.insert(0, commnodedegorder[ID])
                lefts+=1
            elif leftconnections < rightconnections:
                subcommunity_order.append(commnodedegorder[ID])
                rights+=1
            else:
                if lefts < rights:
                    subcommunity_order.insert(0, commnodedegorder[ID])
                    lefts+=1
                elif rights < lefts:
                    subcommunity_order.append(commnodedegorder[ID])
                    rights+=1
                else:
                    r = np.random.rand()
                    if r > 0.5:
                        subcommunity_order.insert(0, commnodedegorder[ID])
                        lefts+=1
                    else:
                        subcommunity_order.append(commnodedegorder[ID])
                        rights+=1
        for (i,node) in enumerate(subcommunity_order):
            angs[node] += 2.*math.pi*i/N           
    return angs

def assign_PSO_radial_coordinates(G, beta):
    """
    Assigns radial coordinates to nodes in the graph G based on their degree and the beta parameter.
    
    Parameters:
    - G: A graph object (from NetworkX) 
    - beta: The popularity fading parameter controlling the distribution of radial coordinates. 
            nodes' radial coords are assigned by their degree rank (ties broken randomly) 
    
    Returns:
    - rads: A numpy array of radial coordinates assigned to the nodes.
    """
    N = len(G.nodes)
    # available radial coords
    radial_coords = beta*2.*np.log(np.linspace(1,N,N))+2.*(1.-beta)*np.log(N)
    sorted_degrees = np.argsort(np.array(G.degree()),axis=0)
    # degree order (rank)
    degree_order = np.flip(sorted_degrees[:,1])
    node_pos = sorted_degrees[:,0]
    rads = np.zeros(N)
    for (cnt,node) in enumerate(degree_order):
        rads[node] = radial_coords[node_pos[cnt]]
    return rads

def assign_angular_coordinates(G, dendogram, nodewise_level = 'degree_greedy', coarsening_method = exponential_coarsening,
                               anchor_num = 1, arrangement_method = 'tsp_christofides', comm_sector_size_prop = 'node_num', seed = None, silent = True):
    
    """
    Assigns angular coordinates to nodes in the graph G based on the provided dendrogram.
    
    Parameters:
    - G: A graph object (from NetworkX) to be embedded 
    - dendogram: multilevel communities, which are arranged on the disk
    - nodewise_level: method to arrange nodes on the lowest-level of the dendrogram
    - coarsening_method: a preweighting scheme that assigns weights to the superedges between communities. 
      Weighted edges must satisfy the triangle inequality! 
    - anchor_num: the number of neighboring communities (from an upper level) which are considered anchor nodes
    - arrangement_method: method to solve the tsp for the arrangement of the communities
    - comm_sector_size_prop: assign angular sectors based on communities' node size or edge size
     
    Returns:
    - angs: A numpy array of angular coordinates assigned to the nodes.
    """
    
    output_dendogram = []
    
    # nodes, number of nodes in the network
    nodes = list(G.nodes()) 
    N = len(nodes)
    max_level = len(dendogram)-1
    # data strucuture conversion for optimised and easier coding: use not only dicts but list of lists as well
    partitions = [partition_at_level(dendogram, level) for level in range(max_level + 1)]
    dendograms = [convert_part_to_comm(dendogram[level]) for level in range(max_level + 1)]
    community_struct = [convert_part_to_comm(partitions[level]) for level in range(max_level + 1)]
    
    if len(community_struct[max_level]) < 2:
        num_communities = len(community_struct[max_level])
        raise ValueError(
            f"Community error: Found only {num_communities} community at the highest level "
            f"(expected at least 2). Please provide another dendrogram as an input parameter."
        )
    
    # handling the anchor_num parameter
    # by definition its value cannot be larger than the number of communities/2 at the highest level
    # however, if a larger value is given then its value is set to its maximal possible value    
    if anchor_num >= len(community_struct[max_level]) // 2:
        max_allowed_anchor_num = max(len(community_struct[max_level]) // 2 - 1, 0)
        anchor_num = max_allowed_anchor_num
        if silent is False:
            warnings.warn(
                f"anchor_num parameter is too large (given: {anchor_num + 1}). "
                f"Adjusted to the maximum allowed value: {max_allowed_anchor_num} "
                f"for a more reasonable result.",
                UserWarning
            )
            
        global clove_parameters
        clove_parameters['anchor_num'] = anchor_num
        
    # angular coordinates
    # init ang coords, each to 0.
    angs = np.zeros(N)
    # extracting communities and arranging them at the highest level of the dendogram
    tsp_G, tsp_order = community_arrange(G, partitions[max_level], coarsening_method = coarsening_method, arrangement_method = arrangement_method, seed = seed)
    # updating angs (not mandatory)
    angs = update_angular_coords(G, tsp_order, community_struct[max_level], angs, comm_sector_size_prop = comm_sector_size_prop)
    # iterating over the different levels of the dendograms in a decreasing order
    for level in range(max_level, 0,-1):
        # current part of the dendogram
        current_dendogram = dendograms[level]
        current_dendogram_len = len(current_dendogram)
        fined_tsp_order = [-1*order_com for order_com in tsp_order]
        # iterating over the individual communities at the current level of the dendro 
        for commID in range(current_dendogram_len):
            if len(current_dendogram[commID]) == 1:
                # commID has only 1 descendant at the next level of the dendogram, 
                # i.e. sub_tsp_order = current_dendogram[commID]
                fined_tsp_order = update_community_order(fined_tsp_order, commID, current_dendogram[commID])  
            else:
                # commID has more than 1 descendant at the next level of the dendogram,
                # arranging the subcommunities more or less in the same way as it is done at the upper level
                # but with two extra 'virtual nodes' representing the communities on the left and right respectively (anchor nodes)
                
                G_sub, subnode2comms = create_sub_net(G, partitions[level-1], community_struct[level], commID, tsp_order, 
                                                      anchor_num = anchor_num) 
                
                sub_tsp_G, sub_tsp_order = community_arrange(G_sub, subnode2comms, coarsening_method = coarsening_method, 
                                                            arrangement_method = arrangement_method, seed = seed)
                if anchor_num > 0:
                    sub_tsp_order = sub_community_chirality(sub_tsp_order)
                # update community order (compared to the previous level)
                fined_tsp_order = update_community_order(fined_tsp_order, commID, sub_tsp_order)
                #print('dend',seed,level, fined_tsp_order,sub_tsp_order)

        output_dendogram = update_output_dendogram(output_dendogram = output_dendogram, level = max_level - level, 
                                                   comms = community_struct[level-1], parts_above = partitions[level])
        
        tsp_order = list(fined_tsp_order)

        #update angular coords
        angs = update_angular_coords(G, tsp_order, community_struct[level-1], angs, comm_sector_size_prop = comm_sector_size_prop)
    # update returning dendrogram
    output_dendogram = update_output_dendogram(output_dendogram = output_dendogram, level = max_level, 
                                                   comms = [[i] for i in range(N)], parts_above = partitions[0])    
    output_dendogram.reverse()
    
    # arranging nodes at the lowest level of dendrogram
    if nodewise_level == 'random_equidistant':
        angs = rand_ang_distribute_lowest_level(lowest_communities = community_struct[0], angs = angs)
        return angs, output_dendogram
            
    elif nodewise_level == 'degree_greedy':
        angs = degreedy_ang_distribute_lowest_level(G = G, lowest_communities = community_struct[0], 
                                                community_order = tsp_order, angs = angs, anchor_num = anchor_num)
        return angs, output_dendogram
    
    elif nodewise_level == 'nodewise_arrangement':    
        angs = nodewise_arrangement_ang_distribute_lowest_level(G = G, lowest_communities = community_struct[0], community_order = tsp_order, 
                                                        angs = angs, arrangement_method = arrangement_method, coarsening_method = coarsening_method,
                                                        comm_sector_size_prop = comm_sector_size_prop, anchor_num = anchor_num, seed = seed)
        return angs, output_dendogram
    else:
        return angs, output_dendogram
def assign_angular_coordinatesLOCMAX(G, local_partitioning = Leiden, nodewise_level = 'degree_greedy', 
                                     coarsening_method = exponential_coarsening, anchor_num = 1, arrangement_method = 'tsp_christofides',
                                     comm_sector_size_prop = 'node_num', cc_decomposition = False, seed = None, silent = True):
    """
    Assigns angular coordinates to nodes in the graph G based iterative local community detection.
    
    Parameters:
    - G: A graph object (from NetworkX) to be embedded 
    - local_partitioning: method for iterative local community detection
    - nodewise_level: method to arrange nodes on the lowest-level of the dendrogram
    - coarsening_method: a preweighting scheme that assigns weights to the superedges between communities. 
      Weighted edges must satisfy the triangle inequality! 
    - anchor_num: the number of neighboring communities (from an upper level) which are considered anchor nodes
    - arrangement_method: method to solve the tsp for the arrangement of the communities
    - comm_sector_size_prop: assign angular sectors based on communities' node size or edge size
    - cc_decomposition: use separate connected components as communities, but only on the uppermost level, 
      lower level communities are detected by local_partitioning  
     
    Returns:
    - angs: A numpy array of angular coordinates assigned to the nodes.
    """
    output_dendogram = []
    nodes = list(G.nodes()) 
    N = len(nodes)
    
    if not cc_decomposition:
        # Use local partitioning if connected components decomposition is disabled
        partition = local_partitioning(G, seed = seed)
    else:
        # Perform connected components partitioning
        partition = connected_components_partitioning(G)

        # Check if there are fewer than 2 connected components
        if len(set(partition.values())) < 2:
            # Raise a warning and fall back to default local partitioning
            if silent is False:
                warnings.warn(
                    "Connected components decomposition is enabled, but only 1 connected component was found. "
                    "Falling back to default local partitioning.", 
                    UserWarning
                )
            partition = local_partitioning(G, seed = seed)

    if not isinstance(partition, dict):
        raise TypeError(
            f"Parameter error: Expected the community finding method to return a dictionary, "
            f"but got {type(partition).__name__} instead. The method should return a dict with nodes as keys "
            f"and corresponding community IDs as values."
            )

    # Proceed to convert the partition to communities after validating the type
    communities = convert_part_to_comm(partition)
    
    if len(communities) < 2:
        num_communities = len(communities)
        raise ValueError(
            f"Community error: The {local_partitioning} method found only {num_communities} community at the highest level "
            f"(expected at least 2). Please provide another community detection method as an input parameter."
        )
        
    # handling the anchor_num parameter
    # by definition its value cannot be larger than the number of communities/2 at the highest level
    # however, if a larger value is given then its value is set to its maximal possible value

    if anchor_num >= len(communities) // 2:
        # Ensure the new anchor_num value is non-negative
        max_anchor_num = max(len(communities) // 2 - 1, 0)
        # Issue a warning and adjust the anchor_num
        if silent is False:
            warnings.warn(
                f"anchor_num parameter is too large: {anchor_num}. "
                f"It has been adjusted to the maximum allowed value: {max_anchor_num} for a reasonable result.",
                UserWarning
            )
            
        # Update the anchor_num to the adjusted value
        anchor_num = max_anchor_num
        
        global clove_parameters
        clove_parameters['anchor_num'] = anchor_num
        
    # init ang coords, each to 0.
    angs = np.zeros(N)    
    # extracting communities and arranging them at the highest level of the dendogram
    tsp_G, tsp_order = community_arrange(G, partition, coarsening_method = coarsening_method, arrangement_method = arrangement_method, seed = seed)   
    
    # update angular coords
    angs = update_angular_coords(G, tsp_order, communities, angs, comm_sector_size_prop = comm_sector_size_prop)
    
    # ini a previous tsp_order variable
    prev_tsp_order = []
    
    # iterate and break down comunities until local_partitioning can identify (resolve) smaller units of communities
    old_unresolvables, new_unresolvables = set([]), set([])
    pseudo_level = 0
    while not tsp_order == prev_tsp_order:
        subcommnum = 0
        fined_tsp_order = [-1*order_com for order_com in tsp_order]
        new_partition = {}        
        for commID in range(len(tsp_order)):
            G_sub_no_virtual = nx.induced_subgraph(G, communities[commID])
            if not commID in old_unresolvables:
                local_subpartition = local_partitioning(G_sub_no_virtual, seed = seed)
            else:
                local_subpartition = {node: 0 for node in communities[commID]}
                
            subpartition = {k: v + subcommnum for k,v in local_subpartition.items()}
            new_partition = new_partition | subpartition
            #if not len(set(subpartition.values())) > 1:
            #    print('before before:', communities[commID])
            
            
            if len(set(subpartition.values())) > 1:
                if anchor_num == 0:
                    sub_tsp_G, sub_tsp_order = community_arrange(G_sub_no_virtual, subpartition, 
                                                   coarsening_method = coarsening_method, arrangement_method = arrangement_method, seed = seed)
                else:
                    G_sub, subnode2comms = create_sub_net(G, subpartition, communities, commID, tsp_order, 
                                                              anchor_num = anchor_num)
                    
                    sub_tsp_G, sub_tsp_order = community_arrange(G_sub, subnode2comms, 
                                                    coarsening_method = coarsening_method,arrangement_method = arrangement_method, seed = seed)
                    sub_tsp_order = sub_community_chirality(sub_tsp_order)

            else:
                # store communities that cannot be resolved anymore, therefore unnecessary to further study 
                # -> they will correspond to the lower level of communities
                unresolvable_ID = list(set(subpartition.values()))[0]
                sub_tsp_order = [unresolvable_ID]
                new_unresolvables.add(unresolvable_ID)


            
            fined_tsp_order = update_community_order(fined_tsp_order, commID, sub_tsp_order)
            subcommnum = max(subpartition.values()) + 1

            
        old_unresolvables, new_unresolvables = set(new_unresolvables), set([])    
        prev_tsp_order = list(tsp_order)
        tsp_order = list(fined_tsp_order)
        
        communities = convert_part_to_comm(new_partition).copy()
        output_dendogram = update_output_dendogram(output_dendogram = output_dendogram, level = pseudo_level, 
                                                   comms = communities, parts_above = partition)
        partition = new_partition.copy()
        
        angs = update_angular_coords(G, tsp_order, communities, angs, comm_sector_size_prop = comm_sector_size_prop)
        pseudo_level+=1
    
    output_dendogram = update_output_dendogram(output_dendogram = output_dendogram, level = pseudo_level, 
                                                   comms = [[i] for i in range(N)], parts_above = partition)    
    output_dendogram.reverse()
    del output_dendogram[1]
 
    # arranging nodes at the lowest level of dendrogram
    if nodewise_level == 'random_equidistant':
        angs = rand_ang_distribute_lowest_level(lowest_communities = communities, angs = angs)
        return angs, output_dendogram

    elif nodewise_level == 'degree_greedy':
        angs = degreedy_ang_distribute_lowest_level(G = G, lowest_communities = communities, community_order = tsp_order, 
                                                    angs = angs, anchor_num = anchor_num)
        return angs, output_dendogram
    elif nodewise_level == 'nodewise_arrangement':
        angs = nodewise_arrangement_ang_distribute_lowest_level(G, lowest_communities = communities , community_order = tsp_order, 
                                                        angs = angs, arrangement_method = arrangement_method, coarsening_method = coarsening_method,
                                                        comm_sector_size_prop = comm_sector_size_prop, anchor_num = anchor_num, seed = seed)
        return angs, output_dendogram
    else:
        return angs, output_dendogram

def build_embedding(rads, angs, int_to_ID, return_cartesian = False):
    """
    building the final data structure to store the embedding coordinates of the nodes either using descartes or polar coords
    """
    N = len(angs)
    if return_cartesian == True:
        return dict(zip([int_to_ID[node] for node in range(N)], (np.vstack((rads*np.cos(angs), rads*np.sin(angs))).T).tolist()))
    else:
        return dict(zip([int_to_ID[node] for node in range(N)], (np.vstack((rads, angs)).T).tolist()))
    
def checking_input_dendogram(N, dendogram):
    if isinstance(dendogram, type(None)):
        return True
    elif isinstance(dendogram, list):
        typesList = [type(dendogram[i]) for i in range(len(dendogram))]
        if not all(element == dict for element in typesList):
            raise Exception("Parameter error: dendrogram is not provided in the correct form. It must be a list of dictionaries.")
        
        def sort_mixed(item):
            try:
                return (0, float(item))  # Numbers come first
            except ValueError:
                return (1, str(item))    # Strings come after
        
        if sorted(set(dendogram[0].keys()), key = sort_mixed) != sorted(range(N)):
            raise ValueError("Parameter error: all nodes must have a community label at the lowest level of the provided dendrogram. Nodes must be indexed as consecutive integers from 0 to N at level zero.")
        max_comm_id_l0 = len(list(set(dendogram[0].values())))
        if sorted(set(dendogram[0].values()), key = sort_mixed) != sorted(range(max_comm_id_l0)):
            raise Exception("Parameter error: community IDs at the lowest level of the dendrogram must be consecutive integers from 0 to n_0, where n_0 is the total number of communities at level zero.")
    else:
        raise Exception("Parameter error: dendrogram is not provided in the correct form. It must be a list of dictionaries.")
    return True

def checking_input_parameters(N, b, deg_fit_sample, dendogram, local_partitioning, nodewise_level, 
                              coarsening_method, anchor_num, arrangement_method, comm_sector_size_prop, 
                              return_cartesian, inplace, k0_decomposition, k1_decomposition, cc_decomposition):
    """
    Validate input parameters for various aspects of community detection and layout algorithms.
    
    Parameters:
    - b : float or None, parameter for popularity fading, must be in [0, 1].
    - deg_fit_sample : sample size for degree fitting.
    - dendogram : hierarchical partitioning information.
    - local_partitioning : local community partitioning method.
    - nodewise_level : method for arranging lower-level communities, must be in predefined list.
    - coarsening_method : method for graph coarsening.
    - anchor_num : float, anchor_num parameter, must be >= 0.
    - arrangement_method : string, a method for solving the TSP for community arrangemenet. Or any other method that returns a valid arrangement.
    - comm_sector_size_prop : string, method to assign angular sectors, must be 'node_num' or 'edge_num'.
    - return_cartesian : bool, indicates whether to return cartesian coordinates.
    - inplace : bool, indicates if operations should be performed in place.
    - k0_decomposition, k1_decomposition, cc_decomposition : bool, indicate if certain decompositions are enabled.
    
    Returns:
    - Calls `checking_input_dendogram` for further dendogram validation.
    """
    
    valid_nodewise_levels = ['degree_greedy', 'random_equidistant', 'nodewise_arrangement', None]
    valid_comm_sector_props = ['edge_num', 'node_num']
    valid_arrangement_methods = ['tsp_greedy', 'tsp_simulated_annealing', 'tsp_threshold_accepting', 'tsp_christofides']
    arrangement_methods_sa = [method + '+sa' for method in valid_arrangement_methods]
    arrangement_methods_ta = [method + '+ta' for method in valid_arrangement_methods]
    
    # Check boolean parameters
    if not isinstance(inplace, bool):
        raise ValueError(f"'inplace' must be a boolean, got {inplace}.")
    
    if not isinstance(return_cartesian, bool):
        raise ValueError(f"'return_cartesian' must be a boolean, got {return_cartesian}.")
    
    # Check 'b' parameter if it's provided
    if b is not None:
        if not (0 <= b <= 1) or np.isnan(b):
            raise ValueError(f"'b' must be in [0, 1], got {b}.")
    
    # Ensure anchor_num is non-negative and integer
    if anchor_num < 0 or not isinstance(anchor_num, int):
        raise ValueError(f"'anchor_num' must be an integer with value >= 0, got {anchor_num}.")
    
    # Validate nodewise_level
    if nodewise_level not in valid_nodewise_levels:
        raise ValueError(f"'nodewise_level' must be one of {valid_nodewise_levels}, got {nodewise_level}.")
    
    # Validate comm_sector_size_prop
    if comm_sector_size_prop not in valid_comm_sector_props:
        raise ValueError(f"'comm_sector_size_prop' must be 'node_num' or 'edge_num', got {comm_sector_size_prop}.")
    
    # Validate arrangement_method (TSP or, instead a custom arrangement method)
    if isinstance(arrangement_method, str):
        valid_methods = valid_arrangement_methods + arrangement_methods_sa + arrangement_methods_ta
        if arrangement_method not in valid_methods:
            raise ValueError(
                f"Invalid 'arrangement_method': {arrangement_method} for TSP. Must be one of {valid_arrangement_methods}, "
                f"or a variation like '+sa' or '+ta', e.g., 'tsp_greedy+sa' or 'tsp_christofides+ta'."
            )
    elif not callable(arrangement_method):
        raise ValueError(
            f"Invalid 'arrangement_method': {arrangement_method}. A custom arrangement method must be callable and return a valid order of communities."
        )

    if not callable(coarsening_method) and not coarsening_method is None:
        raise ValueError(
            f"Invalid 'coarsening_method': {coarsening_method}. It must be callable or None."
        )


    # Check boolean flags for decomposition methods
    for param, name in zip([k0_decomposition, k1_decomposition, cc_decomposition],
                           ['k0_decomposition', 'k1_decomposition', 'cc_decomposition']):
        if not isinstance(param, bool):
            raise ValueError(f"'{name}' must be a boolean, got {param}.")
    
    # Validate dendogram and return result
    return checking_input_dendogram(N, dendogram)

def relabeling_input_parameters(G, dendogram=None):
    """
    Relabels the nodes of graph G with consecutive integer IDs from 0 to N-1.
    Optionally relabels the first level of a given dendogram accordingly.

    Parameters:
    - G : input graph
    - dendogram : hierarchical clustering information (optional)

    Returns:
    - relabeled_G : graph G with relabeled nodes
    - relabeled_dendogram : updated dendogram with relabeled nodes (or None)
    - ID_to_int : dictionary mapping original node IDs to new integer IDs
    - int_to_ID : dictionary mapping new integer IDs back to original node IDs
    """
    N = len(G)  # Number of nodes in the graph
    
    # Create mappings between original node IDs and consecutive integers
    ID_to_int = {ID: idx for idx, ID in enumerate(G.nodes)}
    int_to_ID = {idx: ID for ID, idx in ID_to_int.items()}
    
    # Relabel dendogram if provided
    relabeled_dendogram = (dendogram and
                           [{ID_to_int[ID]: commID for ID, commID in dendogram[0].items()}] +
                           dendogram[1:]) if dendogram else None
    
    # Relabel graph nodes to integers from 0 to N-1
    relabeled_G = nx.relabel_nodes(G, ID_to_int)

    # Ensure relabeling was successful
    if list(relabeled_G.nodes) != list(range(N)):
        raise ValueError("Relabeling Error: node relabeling failed.")

    return relabeled_G, relabeled_dendogram, ID_to_int, int_to_ID

def k01nodes_to_nei(G, k0_decomposition, k1_decomposition):
    """
    Identifies nodes with degree 0 and 1 and returns a dictionary mapping:
    - Nodes with degree 0 to None.
    - Nodes with degree 1 to their single neighbor.

    Parameters:
    G (networkx.Graph): The input graph.
    k0_decomposition (bool): If True, include nodes with degree 0 in the result.
    k1_decomposition (bool): If True, include nodes with degree 1 in the result.

    Returns:
    dict: A dictionary where keys are nodes, and values are either None (for degree 0 nodes)
          or their single neighbor (for degree 1 nodes).
    """
    degrees = dict(G.degree())
    if k0_decomposition == True and k1_decomposition == True:
        k0_dict = dict((node_0, None) for node_0 in degrees.keys() if degrees[node_0]==0)
        k1_dict = dict((node_1, list(G[node_1])[0]) for node_1 in degrees.keys() if degrees[node_1]==1)
        return k0_dict | k1_dict
    elif k0_decomposition == True and k1_decomposition == False:
        k0_dict = dict((node_0, None) for node_0 in degrees.keys() if degrees[node_0]==0)
        return k0_dict
    else:
        k1_dict = dict((node_1, list(G[node_1])[0]) for node_1 in degrees.keys() if degrees[node_1]==1)
        return k1_dict
            
def decompose_k01nodes(G, k0_decomposition, k1_decomposition):
    """
    decompose nodes back based on a decomposition map obtained from k01nodes_to_nei
    """
    G_wout_node01 = G.copy()
    if k0_decomposition == False and k1_decomposition == False:
        return G_wout_node01, None
    else:
        node01s = k01nodes_to_nei(G_wout_node01,k0_decomposition=k0_decomposition, k1_decomposition=k1_decomposition)
        nodes_to_remove = list(node01s.keys())
        G_wout_node01.remove_nodes_from(nodes_to_remove)
        return G_wout_node01, node01s

def recompose_k01nodes(G, angs, ID_to_int, int_to_ID, decomposition_map):
    """
    Recompose nodes with degree 0 or 1 into the graph `G` using the provided `decomposition_map`.
    Decomposition happens at the beginning of the embedding, recomposition does at the end of the embedding.
    
    Parameters:
    G (networkx.Graph): The (already decomposed) graph to which the nodes will be added (back).
    angs (np.array): Array of angular coordinates for the nodes.
    ID_to_int (dict): Mapping from node IDs to integers.
    int_to_ID (dict): Mapping from integers to node IDs.
    decomposition_map (dict): Map of nodes with their corresponding neighbors for re-composition.
                              Degree 1 nodes have neighbors, while degree 0 nodes have None.

    Returns:
    G (networkx.Graph): The updated graph with re-added nodes.
    angs (np.array): Updated angular coordinates for the nodes.
    ID_to_int (dict): Updated ID to integer mapping.
    int_to_ID (dict): Updated integer to ID mapping.
    """
    
    # If no decomposition map is provided, return the original graph and mappings unchanged
    if decomposition_map is None:
        return G, angs, ID_to_int, int_to_ID
    else:
        # Extract all degree 1 nodes and initialize parameters for adding them
        node1s, N, node1_cnt = [node for node in decomposition_map.keys() if not decomposition_map[node] is None], len(G.nodes), 0
        node1_angs = np.zeros(len(node1s))
        rand.shuffle(node1s)
        
        # Recompose degree 1 nodes, while there are many
        while len(node1s) > 0:
            for node1 in node1s:
                # Case 1: The neighbor of the node is already in the graph ( the node is in the giant component)
                if decomposition_map[node1] in ID_to_int.keys():
                    G.add_node(N + node1_cnt)
                    G.add_edge(ID_to_int[decomposition_map[node1]], N + node1_cnt)
                    # inheriting its neigbors ang coordinate
                    node1_angs[node1_cnt] = angs[ID_to_int[decomposition_map[node1]]]
                    # id_to_int and its inverse are updated
                    int_to_ID[N + node1_cnt] = node1
                    ID_to_int[node1] = N + node1_cnt
                    node1_cnt+=1
                    node1s.remove(node1)
                # Case 2: o-o type of connection (node and its single neighbor forms a o-o component)--> both should be recomposed
                elif node1 == decomposition_map[decomposition_map[node1]]:
                    G.add_node(N + node1_cnt)
                    G.add_node(N + node1_cnt + 1)
                    G.add_edge(N + node1_cnt, N + node1_cnt + 1)
                    # in this case node gets a random ang coord and the other node gets the same
                    node1_angs[node1_cnt] = 2.*math.pi*np.random.rand()
                    node1_angs[node1_cnt + 1] = node1_angs[node1_cnt]
                    # id_to_int and its inverse are updated for both nodes (o-o component)
                    int_to_ID[N + node1_cnt] = node1
                    ID_to_int[node1] = N + node1_cnt
                    
                    int_to_ID[N + node1_cnt + 1] = decomposition_map[node1]
                    ID_to_int[decomposition_map[node1]] = N + node1_cnt + 1
                    
                    node1_cnt+=2
                    node1s.remove(decomposition_map[node1])
                    node1s.remove(node1)
                    
        angs = np.hstack((angs,node1_angs)) 
        
        # Recompose degree 0 nodes (isolated nodes)
        node0s = [node for node in decomposition_map.keys() if decomposition_map[node] is None]
        N += node1_cnt
        node0_angs = np.zeros(len(node0s))
        rand.shuffle(node0s)
        for (node0_cnt, node0) in enumerate(node0s):
            G.add_node(N + node0_cnt)
            # assign random ang coordinates for isolated nodes
            node0_angs[node0_cnt] = 2.*math.pi*np.random.rand()
            int_to_ID[N + node0_cnt] = node0
            ID_to_int[node0] = N + node0_cnt
        angs = np.hstack((angs,node0_angs)) 
       
        return G, angs, ID_to_int, int_to_ID
         
def update_output_dendogram(output_dendogram, level, comms, parts_above):
    """
    Update output dendrogram.
    """
    output_dendogram.append({})
    for (i,comm) in enumerate(comms):
        output_dendogram[level][i] = parts_above[comm[0]]
    return output_dendogram

def relabeling_back_output_dendogram(output_dendogram, int_to_ID):
    """
    Relabels the nodes back in the output dedrogram.
    """
    output_dendogram[0] = {int_to_ID[node]: node_label 
                                                 for node, node_label in output_dendogram[0].items()}
    return output_dendogram

def recompose_nodes_to_output_dendogram(output_dendogram, decomposition_map):
    """
    Recomposes nodes to the output dendrogram.
    """
    
    if decomposition_map is None:
        return output_dendogram
    
    node0s = [node for node in decomposition_map.keys() if decomposition_map[node] is None]
    node1s = [node for node in decomposition_map.keys() if not decomposition_map[node] is None]
    
    max_commID_labels = [(None, max(output_dendogram[0].values()))]
    max_commID_labels += [(max(output_dendogram[level].keys()), max(output_dendogram[level].values())) for level in range(1,len(output_dendogram))]
    
    nodewise_level_keys = set(output_dendogram[0].keys())
    nodewise_level_values = set(output_dendogram[0].values())
    
    new_element_in_outdendo = 0
    for node1 in node1s:
        if decomposition_map[node1] in set(output_dendogram[0].keys()):
            output_dendogram[0][node1] = output_dendogram[0][decomposition_map[node1]]
            #print(node1,'o-o')
        else:
            new_element_in_outdendo += 1
            output_dendogram[0][node1] = max_commID_labels[0][1] + new_element_in_outdendo
            
    
    for node0 in node0s:
        new_element_in_outdendo += 1
        output_dendogram[0][node0] = max_commID_labels[0][1] + new_element_in_outdendo
    
    
    new_element_start, new_element_end = max_commID_labels[0][1] + 1, max_commID_labels[0][1] + new_element_in_outdendo + 1
    for level in range(1,len(output_dendogram)):
        new_element_in_outdendo = 0
        for new_element in range(new_element_start, new_element_end):
            new_element_in_outdendo += 1
            output_dendogram[level][new_element] = max_commID_labels[level][1] + new_element_in_outdendo
        new_element_start, new_element_end = max_commID_labels[level][1] + 1, max_commID_labels[level][1] + new_element_in_outdendo + 1
    return output_dendogram 

def embed(G, gamma = None, deg_fit_sample = None, dendrogram = None, local_partitioning = Leiden, 
             nodewise_level = 'degree_greedy', coarsening_method = exponential_coarsening, anchor_num = 1, 
             arrangement_method = 'tsp_christofides+ta', comm_sector_size_prop = 'node_num', k0_decomposition = False, 
             k1_decomposition = False, cc_decomposition = True, return_cartesian = True, inplace = False, 
             rad_assignement = assign_PSO_radial_coordinates,  seed = None, silent = True):

    """
    Embeds a graph G based on: https://arxiv.org/pdf/2410.03270
    
    Parameters:
    - G: Graph (e.g., NetworkX object)
    - gamma: Parameter controlling the decay exponent of the degree distribution
    - deg_fit_sample: Sample size for degree fitting if gamma is not provided
    - dendogram: Optional dendrogram for community structure
    - local_partitioning: Method for local community partitioning (default: Leiden)
    - nodewise_level: Strategy for nodewise optimization (default: 'degree_greedy')
    - coarsening_method: Method to coarsen the graph (default: exponential config)
    - anchor_num: anchor_num parameter for partitioning
    - arrangement_method: Algorithm for TSP (default: 'tsp_christofides')
    - comm_sector_size_prop: Strategy for proportional sector size (default: 'node_num')
    - k0_decomposition: Decompose nodes with degree 0 (default: False)
    - k1_decomposition: Decompose nodes with degree 1 (default: False)
    - cc_decomposition: Use connected component decomposition (default: True)
    - return_cartesian: Return coordinates in Cartesian format (default: False)
    - inplace: Modify G in place by setting node attributes (default: False)
    - rad_assignement: Function to assign radial coordinates (default: assign_PSO_radial_coordinates)
    """
        
    if seed is None:
        seed = np.random.randint(2**31 - 1)
    
    np.random.seed(seed)
    rand.seed(seed)
    
    if gamma is None:
        b = 1./(fit_degree_decay_exponent(G = G, min_samples = deg_fit_sample) - 1)
    else:
        if gamma < 2:
            raise ValueError("Parameter inference error: gamma must be larger than 2.")
        b = 1. / (gamma - 1.) 

    if not dendrogram is None:
        if silent is False:
            warnings.warn(
            "k0, k1, and cc decomposition are enabled only when dendrogram is None. "
            "For safety reasons, these decomposition parameters have been set to False.", 
            UserWarning
            )
        k0_decomposition, k1_decomposition, cc_decomposition = False, False, False


    checking_input_parameters(N = len(G.nodes), b = b, deg_fit_sample = deg_fit_sample, dendogram = dendrogram, local_partitioning = local_partitioning, 
                                 nodewise_level = nodewise_level, coarsening_method = coarsening_method, anchor_num = anchor_num, 
                                 arrangement_method = arrangement_method, comm_sector_size_prop = comm_sector_size_prop, 
                                 return_cartesian = return_cartesian, inplace = inplace, k0_decomposition = k0_decomposition,
                                 k1_decomposition = k1_decomposition, cc_decomposition = cc_decomposition)
    
    
    if local_partitioning in ['Louvain', 'louvain']: local_partitioning = Louvain
    if local_partitioning in ['Leiden', 'leiden']: local_partitioning = Leiden
    
    coarsening_method_STR = (None if coarsening_method is None else 
                              'default function: hypCLOVE.exponential_coarsening' if coarsening_method == exponential_coarsening else 
                              f'custom function: { coarsening_method.__name__}')

    local_partitioning_STR = (None if local_partitioning is None else 
                              'default function: hypCLOVE.Leiden' if local_partitioning == Leiden else 
                              f'custom function: {local_partitioning.__name__}')
    
    rad_assignement_STR = (None if rad_assignement is None else 
                               'default function: hypCLOVE.assign_PSO_radial_coordinates' if rad_assignement == 
                               assign_PSO_radial_coordinates else f'custom function: {rad_assignement.__name__}')
    
    arrangement_method_STR = (arrangement_method if type(arrangement_method) == str 
                              else f'custom function: {arrangement_method.__name__}')
    
    global clove_parameters
    clove_parameters = {
                    'gamma': gamma,
                    'deg_fit_sample': deg_fit_sample,
                    'dendrogram': dendrogram,
                    'local_partitioning': local_partitioning_STR,
                    'nodewise_level': nodewise_level,
                    'coarsening_method': coarsening_method_STR,
                    'anchor_num': anchor_num,
                    'arrangement_method': arrangement_method_STR,
                    'comm_sector_size_prop': comm_sector_size_prop,
                    'k0_decomposition': k0_decomposition,
                    'k1_decomposition': k1_decomposition,
                    'cc_decomposition': cc_decomposition,
                    'return_cartesian': return_cartesian,
                    'inplace': inplace,
                    'rad_assignement': rad_assignement_STR,
                    'seed': seed
                    }
    
    G_preprocessed, decomposition_map = decompose_k01nodes(G, k0_decomposition = k0_decomposition, k1_decomposition = k1_decomposition)
    
    G_copy, dendogram_copy, ID_to_int, int_to_ID = relabeling_input_parameters(G = G_preprocessed, dendogram = dendrogram) 
    nodes = list(G_copy.nodes()) 
    N = len(nodes)    

    # if dendogram is not available
    if dendogram_copy is None:
        # if dendo. is neither available nor required: do level-wise partitioning 
        if not local_partitioning is None:
            # assign angular coords
            angs, output_dendogram = assign_angular_coordinatesLOCMAX(G = G_copy, local_partitioning = local_partitioning, 
                                                    nodewise_level = nodewise_level, coarsening_method = coarsening_method, 
                                                    anchor_num = anchor_num, arrangement_method = arrangement_method,
                                                    comm_sector_size_prop = comm_sector_size_prop, cc_decomposition = cc_decomposition, 
                                                    seed = seed, silent = silent)
            

            # if k01_decomposition == True recompose nodes with degree = 0,1 respectively
            G_copy, angs, ID_to_int, int_to_ID = recompose_k01nodes(G = G_copy, angs = angs, ID_to_int = ID_to_int, 
                                                                   int_to_ID = int_to_ID, decomposition_map = decomposition_map)
            output_dendogram = relabeling_back_output_dendogram(output_dendogram, int_to_ID)
            
            output_dendogram = recompose_nodes_to_output_dendogram(output_dendogram,decomposition_map)
            
            # determining the radial coordinates of the nodes
            rads = rad_assignement(G = G_copy, beta = b)

            coords = build_embedding(rads = rads, angs = angs, int_to_ID = int_to_ID, return_cartesian = return_cartesian)
            if inplace:
                nx.set_node_attributes(G, coords, name='coords')
                return None
            else:
                return {'coords':coords, 'dendrogram': output_dendogram, 'beta_inf': float(b), 'parameters': clove_parameters.copy()}
                
        # if dendogram required but not available
        else:
            dendogram_copy = generate_dendrogram(G_copy, random_state = seed)
            

    # dendogram is provided by the user
    angs, output_dendogram = assign_angular_coordinates(G = G_copy, dendogram = dendogram_copy, nodewise_level = nodewise_level, 
                                      coarsening_method = coarsening_method, anchor_num = anchor_num, arrangement_method =
                                      arrangement_method, comm_sector_size_prop = comm_sector_size_prop, seed = seed, silent = silent)
    
    # if k1_decomposition == True recompose nodes with degree = 1
    G_copy, angs, ID_to_int, int_to_ID = recompose_k01nodes(G = G_copy, angs = angs, ID_to_int = ID_to_int, 
                                                          int_to_ID = int_to_ID, decomposition_map = decomposition_map)
    output_dendogram = relabeling_back_output_dendogram(output_dendogram, int_to_ID)
    output_dendogram = recompose_nodes_to_output_dendogram(output_dendogram,decomposition_map)
   

    # determining the radial coordinates of the nodes
    rads = rad_assignement(G = G_copy, beta = b)
    
    coords = build_embedding(rads = rads, angs = angs, int_to_ID = int_to_ID, return_cartesian = return_cartesian)
    if inplace:
        nx.set_node_attributes(G, coords, name='coords')
        return None
    else:
        return {'coords':coords, 'dendrogram': output_dendogram, 'beta_inf': float(b), 'parameters': clove_parameters.copy()}