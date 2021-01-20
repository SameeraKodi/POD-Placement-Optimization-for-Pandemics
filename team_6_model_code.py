import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd
import csv
import os
from statistics import mean
import matplotlib.pyplot as plt

#################################### Reading dataset files ######################################

# Reading Distance Matrix.csv made by using Google APIs
distance_df = pd.read_csv("Distance Matrix.csv")
distance_df.drop(["Unnamed: 0", "Zip"], axis = 1, inplace = True)

# Removing 'kms' from the distance
for column in distance_df.columns:
    if distance_df[column].dtype == 'object':
        distance_df[column] = distance_df[column].astype('str').str.extract("([-+]?\d*\.\d+|[-+]?\d*\\d+)").astype(float)

# converting into numpy
distance = distance_df.to_numpy()

# Reading POD_capacity.csv made by using schools enrollment number and a multiplication factor 
path = 'POD_capacity.csv'
data = np.genfromtxt(path, dtype=str, delimiter=',', encoding='utf-8-sig')
capacity = data.astype(np.float)

# reading the zip_level_data.csv file made using ACS data
data = pd.read_csv("zip_level_data.csv")
if "Unnamed: 0" in data.columns:
    data.drop(["Unnamed: 0"], axis = 1, inplace = True)

households = np.array(data.households)

# thresholds on population density and making scenarios
thresholds = [13500, 12000, 10000, 7000, 5000, 4000, 3000, 2000, 1000, 0]

for i, t in enumerate(thresholds):
    col = "scenario_" + str(i+1)
    data[col] = data['pop_den'].apply(lambda x: 1 if x >= t else 0)


scenarios = data.iloc[:,5:].to_numpy()
log_scenarios = np.log(scenarios.sum(axis = 0) + 1)

def softmax(scenarios):
    prob = []
    # taking inverse of log of sum of infected zip in each scenario
    for s in scenarios:
        prob.append((1/s)/(1/scenarios).sum())
        
    return prob

probabilities = softmax(log_scenarios)
probabilities = np.asarray(probabilities)


########################################### Model Code ##########################################

############# Model 1 ##############

# setting up ranges
scenes = range(10)
pods = range(47)
zips = range(90)
alpha_values = list(range(2,48))

# Optimizing for the first objective using all scenarios and their probabilities
# Also including the code for global and local conditions

total_distance_travelled = []
global_means = []
local_means = []

for alpha in alpha_values:
    
    m1 = Model("minimize weighted travel distance")
    
    # initiating decision variables

    x = m1.addVars(pods, vtype=GRB.BINARY)
    y = m1.addVars(zips, pods, vtype=GRB.BINARY)
    
    # setting the objective function

    obj = sum(sum(sum(y[j,i] * distance[j,i] * households[j] * scenarios[j,s] * probabilities[s] for j in zips) 
                  for i in pods) for s in scenes)
    m1.setObjective(obj, GRB.MINIMIZE)
    
    # setting the constraints
    
    # opened pods should always be less than alpha
    m1.addConstr(sum(x[i] for i in pods) <= alpha)
    
    # each zip should be assigned to only 1 pod
    for j in zips:
        m1.addConstr(sum(y[j,i] for i in pods) == 1)
    
    # if a pod is being opened, it should have at least one zip assigned
    for i in pods:
        m1.addConstr(sum(y[j,i] * households[j] for j in zips) >= x[i])
    
    # number of households assigned to a pod should be less than or equal to its capacity
    for i in pods:
        m1.addConstr(sum(y[j,i] * households[j] for j in zips) <= x[i]*capacity[i])
        
    m1.optimize()
    
    total_distance_travelled.append(sum(sum(y[j,i].x * distance[j,i] * households[j] for j in zips) for i in pods))
    
    empty_list = []
    for j in zips:
        for i in pods:
            if y[j,i].x == 1:
                empty_list.append((j,i))
                
    distances = []

    for i in empty_list:
        distances.append(distance_df.iloc[i])
        
    global_means.append(mean(distances))
        
    local_distances = []

    for i, x in enumerate(list(scenarios[:,4])):
        if x == 1:
            local_distances.append(distances[i])
            
    local_means.append(mean(local_distances))


# Optimizing for the first objective using each of the scenarios separately without their probabilities
Distances_1 = []

for s in scenes:
    total_distance_travelled_s = []
    
    for alpha in alpha_values:

        m1 = Model("minimize weighted travel distance")

        # initiating decision variables

        x = m1.addVars(pods, vtype=GRB.BINARY)
        y = m1.addVars(zips, pods, vtype=GRB.BINARY)

        # setting the objective function

        obj = sum(sum(y[j,i] * distance[j,i] * households[j] * scenarios[j,s] for j in zips) 
                      for i in pods)
        m1.setObjective(obj, GRB.MINIMIZE)

        # setting the constraints
        
        # opened pods should always be less than alpha
        m1.addConstr(sum(x[i] for i in pods) <= alpha)
        
        # each zip should be assigned to only 1 pod
        for j in zips:
            m1.addConstr(sum(y[j,i] for i in pods) == 1*scenarios[j,s])

        # if a pod is being opened, it should have at least one zip assigned
        for i in pods:
            m1.addConstr(sum(y[j,i] * households[j] for j in zips) >= x[i])
        
        # number of households assigned to a pod should be less than or equal to its capacity
        for i in pods:
            m1.addConstr(sum(y[j,i] * households[j] for j in zips) <= x[i]*capacity[i])

        m1.optimize()

        total_distance_travelled_s.append(sum(sum(y[j,i].x * distance[j,i] * households[j] for j in zips) for i in pods))
        
    Distances_1.append(total_distance_travelled_s)



# plotting the pareto efficiency curve using the distance values got after running above two chunks of code
plt.figure(figsize = (8,6))
plt.plot(alpha_values,total_distance_travelled, label = 'Model1', color="black", zorder = 10)
for i in range(len(Distances_1)):
    
    if i <5:
        leg = 'local_' + str(i+1)
        plt.plot(alpha_values, Distances_1[i], label = leg, linestyle='dashed')
    else:
        leg = 'global_' + str(i+1)
        plt.plot(alpha_values, Distances_1[i], label = leg)

plt.title("Pareto Efficiency graph")
plt.xlabel("# PODs")
plt.ylabel("Total distance travelled by households (Kms)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(np.arange(0,47,2))
plt.savefig('model_1_pareto.png', bbox_inches='tight')

# plotting global vs local plot
plt.figure(figsize = (8,6))
plt.plot(alpha_values,global_means, label = 'Global')
plt.plot(alpha_values, local_means, label = 'Local')
plt.ylabel("Mean distance from ZIP to assigned POD")
plt.xlabel("Max # PODS that can be opened")
plt.title("Mean distances travelled by zips prone to local vs global outbreak")
plt.xticks(np.arange(0,48,5))
plt.legend()
plt.savefig('global_vs_local.png')

############# Model 2 ##############

# Optimizing for the second objective using all scenarios and their probabilities

max_distance = []
    
for alpha in alpha_values:

    m2 = Model("minimize the maximum distance travelled by any household")

    # initiating decision variables

    x = m2.addVars(pods, vtype=GRB.BINARY)
    y = m2.addVars(zips, pods, vtype=GRB.BINARY)
    z = m2.addVars(1, 1, lb = 0.0)
    # setting the objective function

    m2.setObjective(z[0,0], GRB.MINIMIZE)

    # setting the constraints

    # opened pods should always be less than alpha
    m2.addConstr(sum(x[i] for i in pods) <= alpha)
    
    # each zip should be assigned to only 1 pod
    for j in zips:
        m2.addConstr(sum(y[j,i] for i in pods) == 1)
    
    # if a pod is being opened, it should have at least one zip assigned
    for i in pods:
        m2.addConstr(sum(y[j,i] * households[j] for j in zips) >= x[i])
    
    # number of households assigned to a pod should be less than or equal to its capacity
    for i in pods:
        m2.addConstr(sum(y[j,i] * households[j] for j in zips) <= x[i]*capacity[i])

    # z is greater than or equal to the max of all the distances travelled 
    for i in pods:
        for j in zips:
            m2.addConstr(z[0,0] >= distance[j,i]*y[j,i])


    m2.optimize()

    max_distance.append(z[0,0].x)


# Optimizing for the first objective using each of the scenarios separately without their probabilities

Distances_2 = []

for s in scenes:
    max_distance_s = []
    
    for alpha in alpha_values:

        m2 = Model("minimize the maximum distance travelled by any household")

        # initiating decision variables

        x = m2.addVars(pods, vtype=GRB.BINARY)
        y = m2.addVars(zips, pods, vtype=GRB.BINARY)
        z = m2.addVars(1, 1, lb = 0.0)
        # setting the objective function

        m2.setObjective(z[0,0], GRB.MINIMIZE)

        # setting the constraints

        m2.addConstr(sum(x[i] for i in pods) <= alpha)

        for j in zips:
            m2.addConstr(sum(y[j,i] for i in pods) == 1*scenarios[j,s])

        for i in pods:
            m2.addConstr(sum(y[j,i] * households[j] for j in zips) <= x[i]*capacity[i])
        
        for i in pods:
            m2.addConstr(sum(y[j,i] for j in zips) >= x[i])
            
        # z is greater than or equal to the max of all the distances travelled 
        for i in pods:
            for j in zips:
                m2.addConstr(z[0,0] >= distance[j,i]*y[j,i])


        m2.optimize()

        max_distance_s.append(z[0,0].x)
        
    Distances_2.append(max_distance_s)

# plotting the pareto efficiency curve using the distance values got after running above two chunks of code
plt.figure(figsize = (8,6))
plt.plot(alpha_values,max_distance, label = 'Model2', color="black", zorder = 10)
for i in range(len(Distances_2)):
    
    if i <5:
        leg = 'local_' + str(i+1)
        plt.plot(alpha_values, Distances_2[i], label = leg, linestyle='dashed')
    else:
        leg = 'global_' + str(i+1)
        plt.plot(alpha_values, Distances_2[i], label = leg)

plt.title("Pareto Efficiency graph")
plt.xlabel("# PODs")
plt.ylabel("Total distance travelled by households (Kms)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(np.arange(0,47,2))
plt.savefig('model_2_pareto.png', bbox_inches='tight')

################ Code End ##################