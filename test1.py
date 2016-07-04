# -*- coding: utf-8 -*-
 
import pandas as pd
import math
import networkx as nx
from collections import defaultdict


def cal_distance(lng1,lat1,lng2,lat2):
    R=6378137
    d_lat=(lat1-lat2)/2
    d_lng=(lng1-lng2)/2
    dis=2*R*math.asin(math.sqrt((math.sin(math.pi/180*d_lat))**2+math.cos(math.pi/180*lat1)*math.cos(math.pi/180*lat2)*(math.sin(math.pi/180*d_lng))**2))
    return dis
'''
for i in range(len(a)):
    for j in range(len(a)):
        l.append([i,j,cal_distance(a['Lat'][i],a['Lng'][i],a['Lat'][j],a['Lng'][j])])
'''  

#read data
G=nx.DiGraph()


data_base_path = "D:/Ali/lastmile/data/"
A = pd.read_csv(data_base_path+"1.csv",header=0)
B = pd.read_csv(data_base_path+"2.csv",header=0).rename(columns={'Lng':'Lng_Drop','Lat':'Lat_Drop'})
S = pd.read_csv(data_base_path+"3.csv",header=0).rename(columns={'Lng':'Lng_Pick','Lat':'Lat_Pick'})
E = pd.read_csv(data_base_path+"5.csv",header=0,nrows=20)
E=pd.merge(E,S,on=['Shop_id'],how='left')
E=pd.merge(E,B,on=['Spot_id'],how='left')
#genarate graph
for i in range(len(A)):
    G.add_node(A['Site_id'][i],Lng=A['Lng'][i],Lat=A['Lat'][i],tp='Start')
    
for i in range(len(E)):
    G.add_node(E['Order_id'][i]+'_Pick',tp='pickup',Lng=E['Lng_Pick'][i],Lat=E['Lat_Pick'][i],quantity=E['Num'][i],
               P_time=E['Pickup_time'][i],D_time=E['Delivery_time'][i],pair=E['Order_id'][i]+'_Drop')
    G.add_node(E['Order_id'][i]+'_Drop',tp='dropoff',Lng=E['Lng_Drop'][i],Lat=E['Lat_Drop'][i],quantity=E['Num'][i],
               P_time=E['Pickup_time'][i],D_time=E['Delivery_time'][i],pair=E['Order_id'][i]+'_Pick')
 


        

def tabu_search_vrpcd(G, cross_dock, Q, T, load, px='Lng', py='Lat',
                      node_type='tp', quantity='quantity', pair='pair',
                      cons='consolidation', tolerance=20, L=12, k=2, a=10,
                      diversification_iter=0):

    # Calculate distance for every pair of nodes and save it in a dictionary.
    dist = defaultdict(dict)
    for u in G:
        for v in G:
            dist[u][v] = cal_distance(G.node[u][px],G.node[u][py],G.node[v][px],G.node[v][py])
            
    #generate initial solution
    if G.number_of_edges() == 0:
        vehicle_id = 0
        capacity = {}
        duration = {}
        for u in G:
            if G.node[u][node_type] == 'pickup':
                capacity[vehicle_id] = [G.node[u][quantity],
                                        G.node[u][quantity]]
                duration[vehicle_id] = 2 * dist[cross_dock][u]\
                                       + 2 * dist[cross_dock][G.node[u][pair]]
                vehicle_id += 1

        # Construct an initial-primitive solution.
        G = construct_primitive_solution(G, cross_dock, node_type, pair, dist)
        used_vehicles = sum(1 for vehicle in capacity.keys() if
                            capacity[vehicle][0] != 0 and capacity[vehicle][1] != 0)
    frequency = {(u, v): 1 if G.has_edge(u, v) else 0 for u in G for v in G}
    pickup_tour = {i: sum(data['weight'] for u, v, data in G.edges(data=True)
                          if data[node_type] == 'pickup' and data['vehicle'] == i)
                   for i in duration.keys()}

    cost = sum(duration.values())
    best_cost = float('inf')
    best_sol = None
    tabu_list = []
    max_iter = 0
    diversification_process = False
    diver_iter = 0
    diversification_iter = tolerance if diversification_iter is None \
        else diversification_iter

    # Tabu algorithm's main loop.
    while max_iter < tolerance:

        try:
            cost = _apply_move(G, cross_dock, cost, best_cost, dist,
                               tabu_list, capacity, duration, frequency,
                               pickup_tour, load, Q, T, L, node_type, quantity,
                               cons, pair, used_vehicles, k, a,
                               diversification_process)
        except IOError:
            break
        if cost < best_cost:
            max_iter = 0
            best_cost = cost
            best_sol = G.copy()
        else:
            if not diversification_process:
                max_iter += 1
        # Count diversification iterations.
        if diversification_process:
            diver_iter += 1
            if diver_iter == diversification_iter:
                diversification_process = False

        # Activate diversification process.
        if max_iter == tolerance and diver_iter == 0:
            if not diversification_process and diversification_iter != 0:
                max_iter = 0
            diversification_process = True

    return best_sol, best_cost
    
    
    
    

def construct_primitive_solution(G, cross_dock, node_type, pair, dist):
    """
    Constructs a fist solution for Clarke-Wright algorithm.

    This solution uses for every pair of nodes (supplier-customer) a different
    vehicle to serve them.

    :param G: graph, a graph with nodes which represents suppliers and customers.
    :param cross_dock: label of node which represents cross-dock.
    :param node_type: string , node data key corresponding to the type of node
    (supplier or customer).
    :param pair: string, node data key corresponding to the label of node's pair.
    :param dist: array of distance between every pair of nodes.
    """
    vehicle_id = 0
    for u in G:
        if G.node[u][node_type] == 'pickup':
            pair_node = G.node[u][pair]
            G.node[u]['vehicle'] = vehicle_id
            G.node[pair_node]['vehicle'] = vehicle_id
            # find the best start point 
            G.add_edge(cross_dock, u, weight=dist[cross_dock][u],
                       type=G.node[u][node_type], vehicle=vehicle_id)
            G.add_edge(u, cross_dock, weight=dist[u][cross_dock],
                       type=G.node[u][node_type], vehicle=vehicle_id)
            G.add_edge(cross_dock, pair_node,
                       weight=dist[cross_dock][pair_node],
                       type=G.node[pair_node][node_type], vehicle=vehicle_id)
            G.add_edge(pair_node, cross_dock,
                       weight=dist[pair_node][cross_dock],
                       type=G.node[pair_node][node_type], vehicle=vehicle_id)
            vehicle_id += 1
    return G
