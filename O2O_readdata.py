# -*- coding: utf-8 -*-
 
import pandas as pd
import math
import networkx as nx



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
    
G=nx.DiGraph()


data_base_path = "D:/Ali/lastmile/data/"
A = pd.read_csv(data_base_path+"1.csv",header=0)
B = pd.read_csv(data_base_path+"2.csv",header=0)
S = pd.read_csv(data_base_path+"3.csv",header=0)
E = pd.read_csv(data_base_path+"5.csv",header=0,nrows=30)


for i in range(len(A)):
    G.add_node(A['Site_id'][i],lng=A['Lng'][i],lat=A['Lat'][i],tp='Start')
    G.add_node(B['Spot_id'][i],lng=A['Lng'][i],lat=A['Lat'][i],tp='Dropoff')
    G.add_node(S['Shop_id'][i],lng=A['Lng'][i],lat=A['Lat'][i],tp='Pickup')




