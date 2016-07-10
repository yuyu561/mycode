# -*- coding: utf-8 -*-
import itertools
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
		routing_list={}

        # Construct an initial-primitive solution.
        G, capacity, duration = construct_primitive_solution(G, cross_dock, node_type, pair, dist)
		
		
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
    
    
    
    

def construct_primitive_solution(G, capacity, duration, routing_list, node_type, pair, dist):
    vehicle_id = 0
    for u in G:
        if G.node[u][node_type] == 'pickup':
            pair_node = G.node[u][pair]
            G.node[u]['vehicle'] = vehicle_id
            G.node[pair_node]['vehicle'] = vehicle_id

            # find the best start point 
			min_dist=float('inf')
			for v in G:
				if G.node[v][node_type] == 'start':
					now_dist=dist[u][v]
					if now_dist < min_dist :
						start_node,min_dist = (v,now_dist) 

            G.add_edge(start_node, u, weight=dist[start_node][u],
                       type=G.node[u][node_type], vehicle=vehicle_id)
            G.add_edge(u, pair_node, weight=dist[u][pair_node],
                       type=G.node[pair_node][node_type], vehicle=vehicle_id)

			routing_list[vehicle_id] = [start_node, u, pair_node]
			duration[vehicle_id] = dist[start_node][u] + dist[u][pair_node]
			capacity[vehicle_id] = [                       ]
			
            vehicle_id += 1
    return G, capacity, duration

def calculate_route_cost(route, dist):
	route_distance=0
	for i in range(1,len(route)):
		route_distance+=dist[route(i-1)][route(i)]
	return route_distance
	
def calculate_route_capacity(G, route, quantity):
	route_capacity = {}
	cumul_c = 0
	for u in route:
		if G.node[u][node_type] == 'pickup':
			cumul_c += G.node[u][quantity]
        if G.node[u][node_type] == 'dropoff':
            cumul_c -= G.node[u][quantity]
        route_capacity[u] = cumul_c
	return route_capacity
    
def check_route_capacity(route_capacity,capacity_cap):
    return max([route_capacity[i] for i in route_capacity])>capacity_cap
    
def check_route_order(G, route):
    flag=[]
	for uu in route:
		if (G.node[uu][node_type] == 'pickup' and route.index(uu)<route.index(G.node[uu][pair]))
			or (G.node[uu][node_type] == 'dropoff' and route.index(uu)>route.index(G.node[uu][pair]))
            or (G.node[uu][node_type] == 'start' and route.index(uu)==0):
            flag.append(0)
        else:
            flag.append(1)
    return sum(flag)==0

def find_vehicle_by_node(routing_list, u):
	for vehicle_id in routing_list:
		if u in routing_list[vehicle_id]:
			vehicle = vehicle_id
    return vehicle 
    
def clarke_wright(G, capacity, duration, routing_list, dist, node_type,
                  quantity, pair, Q, T):
    savings = {}
    for u in G:
        vehicle_id = find_vehicle_by_node(routing_list, u)  
        current_route=routing_list[vehicle_id]
        current_cost_u=calculate_route_cost(current_route, dist)
        for v in G:
            if G.node[u][node_type] == 'pickup' and G.node[v][node_type] == 'pickup':
                pair_u = G.node[u][pair]
                pair_v = G.node[v][pair]
        
        
				start_node=current_route[0]
				current_route.pop(0)
				current_route.append(v)
				current_route.append(pair_v)
				
                best_route, best_route_cost = [], float('inf')
				for route in itertools.permutations(current_route):
                    route.insert(0,start_node)
					#check if it is feasible
                    route_capacity = calculate_route_capacity(G, route, quantity)
                    route_distance = calculate_route_cost(route, dist)
                    if not(check_route_order and check_route_capacity):
                        continue
                    if route_distance<best_route_cost:
                        best_route, best_route_cost=route, route_distance
                if not best_route_cost == float('inf'):
                    savings[(u,v)] = current_cost_u + cost_v
                    
                        
                    

				
				
                savings[(u, v)] = dist[cross_dock][u] + dist[cross_dock][v] \
                                  - dist[u][v] + dist[cross_dock][pair_u] \
                                  + dist[cross_dock][pair_v] - dist[pair_u][pair_v]

    import operator

    sorted_savings = sorted(savings.items(), key=operator.itemgetter(1),
                            reverse=True)

    for (u, v), savings in sorted_savings:
        pair_u = G.node[u][pair]
        pair_v = G.node[v][pair]
        if (G.has_edge(cross_dock, u)) and (G.has_edge(v, cross_dock))\
                and (not G.has_edge(u, v)):
            index, pair_index = (0, 1) if G.node[v][node_type] == 'pickup' else (1, 0)

            # Check if vehicle's capacity is sufficient to service new node.
            if capacity[G.node[v]['vehicle']][index] + G.node[u][quantity] > Q \
                    and capacity[G.node[pair_v]['vehicle']][pair_index] + \
                            G.node[u][quantity] > Q:
                continue

            # Initialize object to check the feasibility of solution
            # based on its duration.
            if duration[G.node[v]['vehicle']] + (
                    dist[pair_u][pair_v] + dist[u][v] - dist[cross_dock][v]) - (
                    dist[cross_dock][pair_v] + dist[cross_dock][u]
                    + dist[cross_dock][pair_u]) > T:
                continue

            if not G.has_edge(u, cross_dock) or not G.has_edge(cross_dock, v) \
                    or not G.has_edge(pair_u, cross_dock) or not G.has_edge(
                    cross_dock, pair_v):
                continue

            # Update capacity of vehicles.
            capacity[G.node[v]['vehicle']][index] += G.node[u][quantity]
            capacity[G.node[v]['vehicle']][pair_index] += G.node[u][quantity]
            capacity[G.node[u]['vehicle']][index] -= G.node[u][quantity]
            capacity[G.node[u]['vehicle']][pair_index] -= G.node[u][quantity]

            # Update duration of vehicles.
            duration[G.node[v]['vehicle']] += (
                dist[pair_u][pair_v] + dist[u][v] - dist[cross_dock][v]
                - dist[cross_dock][pair_v] + dist[cross_dock][u] + dist[cross_dock][pair_u])
            duration[G.node[u]['vehicle']] -= (
                dist[u][cross_dock] + dist[cross_dock][u]) + (
                dist[pair_u][cross_dock] + dist[cross_dock][pair_u])

            # Update solution.
            G.node[u]['vehicle'] = G.node[v]['vehicle']
            G.node[pair_u]['vehicle'] = G.node[v]['vehicle']
            G.remove_edge(u, cross_dock)
            G.remove_edge(cross_dock, v)
            G.edge[cross_dock][u]['vehicle'] = G.node[v]['vehicle']
            G.edge[cross_dock][pair_u]['vehicle'] = G.node[v]['vehicle']
            G.add_edge(u, v, weight=dist[u][v], type=G.node[v][node_type],
                       vehicle=G.node[v]['vehicle'])
            G.remove_edge(pair_u, cross_dock)
            G.remove_edge(cross_dock, pair_v)
            G.add_edge(pair_u, pair_v, weight=dist[pair_u][pair_v],
                       type=G.node[pair_v][node_type], vehicle=G.node[v]['vehicle'])
    return G