#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 03:42:07 2020

@author: shukaif
"""

import networkx
import numpy



def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    # If no base graph is provided, generate a random preferential attachment power law graph as a starting point.
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = networkx.barabasi_albert_graph(n=n, m=m)

    # We modify the graph by probabilistically dropping some edges from each node. 
    for node in graph:
        neighbors = list(graph[node].keys())
        if(len(neighbors) > 0):
            quarantineEdgeNum = int( max(min(numpy.random.exponential(scale=scale, size=1), len(neighbors)), min_num_edges) )
            quarantineKeepNeighbors = numpy.random.choice(neighbors, size=quarantineEdgeNum, replace=False)
            for neighbor in neighbors:
                if(neighbor not in quarantineKeepNeighbors):
                    graph.remove_edge(node, neighbor)
    
    return graph