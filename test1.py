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
    #if G.number_of_edges() == 0:
    vehicle_id = 0
	routing_dic={}

    # Construct an initial-primitive solution.
    routing_dic = construct_primitive_solution(G, routing_dic, node_type, pair, dist)
	routing_dic = clarke_wright(G, routing_dic, dist, node_type, quantity, pair)
	
	
    used_vehicles = len(routing_dic)

    cost = calculate_routing_dic_cost(G, routing_dic,dist)
    best_cost = float('inf')
    best_sol = routing_dic
    tabu_list = []
    diver_iter_total = 0
    diver_iter = 0
    diversification_iter = tolerance if diversification_iter is None \
        else diversification_iter
	new_best = 1
	
    # Tabu algorithm's main loop.
    while diver_iter_total < tolerance:
		if new_best ==1 or diver_iter%10==0:
			
			temp_solution, cost=inter_route_change()
			new_best=0
		else:
			temp_solution, cost=intra_route_change()
		
        if cost < best_cost and check_temp_solution_feasible(temp_solution):
            diver_iter = 0
            best_cost = cost
            best_sol = temp_solution
			tabu_list = []
        else:
			diver_iter += 1
        diver_iter_total += 1
 
    return best_sol, best_cost
    
    
def calculate_routing_dic_cost(G, routing_dic,dist):
	return sum([calculate_route_duration(G,routing_dic[vehicle_id],dist) for vehicle_id in routing_dic])
		
 
    

def construct_primitive_solution(G, routing_dic, node_type, pair, dist):
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

			routing_dic[vehicle_id] = [start_node, u, pair_node]
            vehicle_id += 1
    return routing_dic
'''
def calculate_route_cost(route, dist):
	route_distance=0
	if len(route)>1:
		for i in range(1,len(route)):
			route_distance+=dist[route[i-1]][route[i]]
	return route_distance
'''
def calculate_route_capacity(G, route, quantity,node_type):
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

def calculate_route_duration(G,route,dist):
	P_time='P_time'
	D_time='D_time'
	node_type='tp'
	route_timeline={}
	arrive_time, leave_time, penalty_time =0, 0, 0
	route_timeline[route[0]] = [arrive_time, leave_time]
	if len(route)>1:
		for i in range(1,len(route)):
			u = route[i]
			drive_time = round(dist[route[i-1]][route[i]]/250)
			arrive_time = drive_time + leave_time
			if G.node[u][node_type] == 'pickup':
				if  G.node[u][P_time] > arrive_time:
					leave_time = G.node[u][P_time]
				else:
					leave_time = arrive_time
					penalty_time += (arrive_time - G.node[u][P_time]) * 5
			if G.node[u][node_type] == 'dropoff':
				service_time = round(3*math.sqrt(G.node[u][quantity]) + 5)
				if  G.node[u][D_time] >  arrive_time:
					leave_time = service_time + arrive_time
				else:
					leave_time = service_time + arrive_time
					penalty_time += (arrive_time - G.node[u][D_time]) * 5
			route_timeline[u] = [arrive_time, leave_time]
	route_duration = penalty + leave_time
	if leave_time >=720:
		route_duration = float('inf')
	return route_duration
	
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

def find_vehicle_by_node(routing_dic, u):
	for vehicle_id in routing_dic:
		if u in routing_dic[vehicle_id]:
			vehicle = vehicle_id
    return vehicle 
    
def clarke_wright(G, routing_dic, dist, node_type, quantity, pair):
    savings = {}
    for u in G:
        vehicle_id_u = find_vehicle_by_node(routing_dic, u)  
        current_route=routing_dic[vehicle_id_u]
        current_cost_u=calculate_route_duration(G,current_route,dist)
        for v in G:
            if G.node[u][node_type] == 'pickup' and G.node[v][node_type] == 'pickup':
                pair_u = G.node[u][pair]
                pair_v = G.node[v][pair]
        
				vehicle_id_v = find_vehicle_by_node(routing_dic, v)  
				current_route_v=routing_dic[vehicle_id_v]
				current_cost_v=calculate_route_duration(G,current_route_v,dist)
				
				start_node=current_route[0]
				current_route.pop(0)
				current_route.append(v)
				current_route.append(pair_v)
				
                best_route, best_route_cost = [], float('inf')
				for route in itertools.permutations(current_route):
                    route.insert(0,start_node)
					#check if it is feasible
                    route_capacity = calculate_route_capacity(G, route, quantity,node_type)
                    route_distance = calculate_route_duration(G,route,dist)
                    if not(check_route_order and check_route_capacity):
                        continue
                    if route_distance<best_route_cost:
                        best_route, best_route_cost=route, route_distance
                if not best_route_cost == float('inf'):
					current_route_v.remove(v)
					current_route_v.remove(pair_v)
					reduced_route_v_cost=calculate_route_duration(G,current_route_v,dist)
                    
					savings[(u,v)] = current_cost_u + current_cost_v - best_route_cost - reduced_route_v_cost
 
    import operator

    sorted_savings = sorted(savings.items(), key=operator.itemgetter(1),
                            reverse=True)

    for (u, v), savings in sorted_savings:
        pair_u = G.node[u][pair]
        pair_v = G.node[v][pair]
		vehicle_id_u = find_vehicle_by_node(routing_dic, u) 
		vehicle_id_v = find_vehicle_by_node(routing_dic, v) 
        if  vehicle_id_u != vehicle_id_v and G.node[u][node_type] == 'pickup' and G.node[v][node_type] == 'pickup':
			urrent_route=routing_dic[vehicle_id_u]
			current_cost_u=calculate_route_duration(G,urrent_route,dist)
			current_route_v=routing_dic[vehicle_id_v]
			current_cost_v=calculate_route_duration(G,current_route_v,dist)
			
			start_node=current_route[0]
			current_route.pop(0)
			current_route.append(v)
			current_route.append(pair_v)
			
            best_route, best_route_cost = [], float('inf')
			for route in itertools.permutations(current_route):
                route.insert(0,start_node)
				#check if it is feasible
                route_capacity = calculate_route_capacity(G, route, quantity,node_type)
                route_distance = calculate_route_duration(G,route,dist)
                if not(check_route_order and check_route_capacity):
                    continue
                if route_distance<best_route_cost:
                    best_route, best_route_cost=route, route_distance
            if not best_route_cost == float('inf'):
				current_route_v.remove(v)
				current_route_v.remove(pair_v)
				if len(current_route_v)<=1:
					routing_dic.pop(vehicle_id_v)
				else:
					routing_dic[vehicle_id_v] = current_route_v
				routing_dic[vehicle_id_u] = best_route
				
    return routing_dic