# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:14:54 2015

@author: mark_kamper
"""
import numpy as np
import networkx as nx

def googleMatrix(links, alpha):
    #Initializing network matrix 
    net_size = len(links)
    network_matrix = np.zeros((net_size, net_size))
    #Getting links in each node
    for i in range(net_size):
        if len(links[i]) == 0:
            network_matrix[i,:] = 1/net_size
        else:
            for link in links[i]:
                network_matrix[i, link] = 1/len(links[i]) 
    #Transposing matrix
    network_matrix = np.transpose(network_matrix)         
    #Adding random jumping to the mix
    random_jump = np.array([[1/net_size] for i in range(net_size)])         
    
    #Returning google matrix
    google_matrix = alpha*network_matrix + (1-alpha)*random_jump
    return google_matrix
    
    
def pageRank(google_matrix, epsilon):
    net_size = np.shape(google_matrix)[0]
    #Initializing eigenvector
    pageRank = np.zeros(net_size)
    pageRank_old = np.zeros(net_size)
    pageRank[0] = 1
    #Performing power calculation until convergence criterion is met 
    while np.linalg.norm(pageRank - pageRank_old) > epsilon:
        pageRank_old = np.copy(pageRank)
        pageRank = np.dot(google_matrix, pageRank)
        #Normalizing for numeric stability
    return pageRank/np.sum(pageRank)
        
        

print(pageRank(googleMatrix([[],[0,2,3],[0,3],[0,2]], 0.85), 0.01))
            
            
network = nx.DiGraph()
network.add_edges_from([(1,2),(1,3),(1,0),(2,3),(2,0),(3,2),(3,0)])
nx.draw(network)
print(nx.pagerank(network, alpha = 0.85))



                
            
            
    