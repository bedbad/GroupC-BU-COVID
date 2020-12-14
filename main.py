#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 03:50:20 2020

@author: shukaif
"""

# module load python3/3.7.9
# pip install --user networkx

import networkx
from SEIRmodel import SEIRmodel
import sys
from lib import custom_exponential_graph


numNodes = 10000
baseGraph = networkx.barabasi_albert_graph(n=numNodes, m=9)
network = custom_exponential_graph(baseGraph, scale = 100)

# parameter adjust can be found https://github.com/ryansmcgee/seirsplus/wiki/ExtSEIRSNetworkModel-class
model = SEIRmodel(G=network, beta=0.45, sigma=1/3.1, lamda=1/2.2, gamma=1/6.5, initE=100)

model.run(T=300)
# test.test()