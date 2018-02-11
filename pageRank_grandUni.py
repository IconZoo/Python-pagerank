# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:04:41 2015

This pagerank implementation unifies the google-matrix and pagerank functions into one

@author: Mark Kamper Svendsen
"""
# Imports 
import numpy as np
import time


#Initializing default values for max number of computation steps convergence and alpha
def pagerank(links, alpha = 0.85, epsilon = 0.01, maxsteps = 200):
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
    
    #Computing pageRank:
    #Initializing eigenvector
    pagerank = np.zeros(net_size)
    pagerank_old = np.zeros(net_size)
    pagerank[0] = 1
    step = 0
    #Measuring calculation time
    start_time = time.time()
    
    #Performing power calculation until convergence criterion is met or max iteration is reached 
    while np.linalg.norm(pagerank - pagerank_old) > epsilon and step < maxsteps:
        pagerank_old = np.copy(pagerank)
        pagerank = np.dot(google_matrix, pagerank)
        step += 1
    
    end_time = time.time()
    calc_time = end_time - start_time
    
    if np.linalg.norm(pagerank - pagerank_old) > epsilon:
        print("[-] The convergence criterion was not reached :-(")
        print("[*] The pagerank calculations converged to within: " + str(np.linalg.norm(pagerank - pagerank_old)))
        print("[*] The calculation was performed in: " + str(calc_time))
    else:
        print("[+] Pagerank was succesfully calculated to within the convergence criterion of " + str(epsilon) + " in " + str(step) + " steps and " + str(calc_time) + " seconds :-D")
    #Normalizing for numeric stability
    return pagerank/np.sum(pagerank)
    
print(pagerank([[],[0,2,3],[0,3],[0,2]], alpha = 0.9 ,epsilon = 1e-6))
            