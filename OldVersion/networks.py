#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:12:14 2020

@author: delainerogers
"""

import snap
import numpy as np

def create_connections(graph, id_range_1, id_range_2, connections):
    for i in id_range_1:
        network = np.random.choice(id_range_2, connections)
        for j in network:
            graph.AddEdge(i, int(j))

def create_subpopulation(graph, people, id_range, connections, role, age_range):
    for i in id_range:
        person = {}
        person["age"] = np.random.choice(age_range)
        person["role"] = role
        person["id"] = i
        people.append(person)
        graph.AddNode(i)
    create_connections(graph, id_range, id_range, connections)
            
def generate_network(undergrad_connections=11, 
                     grad_connections=11, 
                     faculty_connections=5,
                     staff_connections=20, 
                     undergrad_to_grad=5, 
                     undergrad_to_faculty=3,
                     grad_to_faculty=5, 
                     staff_to_undergrad=20,
                     staff_to_grad=15,
                     staff_to_faculty=5):
    
    scaling_factor = 0.001
    undergrads = int(17000 * scaling_factor)
    grads = int(15000 * scaling_factor)
    faculty = int(4000 * scaling_factor)
    staff = int(6000 * scaling_factor)
    total = undergrads + grads + faculty + staff
    
    # Generate lists of ids
    start_id = 0
    undergrad_range = range(undergrads)
    start_id += undergrads
    grad_range = range(start_id, start_id + grads)
    start_id += grads
    faculty_range = range(start_id, start_id + faculty)
    start_id += faculty
    staff_range = range(start_id, start_id + staff)

    people = []
    
    #G2 = snap.GenRndGnm(snap.PUNGraph, 100, 1000)
    G1 = snap.TUNGraph.New()

    # Create undergrad population and their connections
    create_subpopulation(G1, people, undergrad_range, undergrad_connections, "undergrad", range(17,23))
    
    # Create grad population and their connections
    create_subpopulation(G1, people, grad_range, grad_connections, "grad", range(21,30))
    
    # Create faculty population and their connections
    create_subpopulation(G1, people, faculty_range, faculty_connections, "faculty", range(30,70))
    
    # Create staff population and their connections
    create_subpopulation(G1, people, staff_range, staff_connections, "staff", range(18,70))
    
    # Create connections between different populations
    # Undergrad to grad
    create_connections(G1, undergrad_range, grad_range, undergrad_to_grad)
    
    # Undergrad to faculty
    create_connections(G1, undergrad_range, faculty_range, undergrad_to_faculty)
    
    # Grad to faculty
    create_connections(G1, grad_range, faculty_range, grad_to_faculty)
    
    # Staff to everyone
    create_connections(G1, staff_range, undergrad_range, staff_to_undergrad)
    create_connections(G1, staff_range, grad_range, staff_to_grad)
    create_connections(G1, staff_range, faculty_range, staff_to_faculty)
    
    return G1, people
   
