# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:06:27 2021

@author: Cristian Domínguez Cachinero
Code of the Master's Thesis "Routing Drones in Smart Cities: Heuristic Solutions"
"""

'Libraries'
import numpy as np
import math as mth
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum

'Initial settings'
rnd = np.random
rnd.seed(67) #To have the same random values

n0 = 50 #number of customers
xlength = 2000 #x longitude of the covered area of service [m]
ylength = 2000 #y longitude of the covered area of service [m]
xc = rnd.random(n0+1)*xlength #x coordinates of the address of the customers and the depot
yc = rnd.random(n0+1)*ylength #y coordinates of the addrees of the customers and the depot

#Dominion parameters
N0 = [i for i in range(1, n0+1)] #Set of number of customers' address
N = [0] + N0 #Set of number of customer's address + depot
A = [(i,j) for i in N for j in N if i != j] #Set of possible travelling arcs
d = {(i,j): np.hypot(xc[i]-xc[j],yc[i]-yc[j]) for i, j in A} #Distance of each arc [m]
Kub = 10 ** 15 #Upper bound constant
tau = 120 #Time for the drone to descend, deliver/pick package and ascend [s] / BETTER: ascend and descend
v = 15 #Speed of the drone [m/s]
T = 43200 #Time limit (12h) [s]
q = {i: rnd.uniform(0,2) for i in N0} #To-deliver parcel weight (for each customer's address) [kg] ###uniform por randint
Q = 2  #Weight capacity of a drone [kg]
B = 100000 #Budget [€ or other financial unit]
droneC = 1500 #Cost of a delivery drone [same financial unit that budget]
batteryC = 70 #Cost of a drone's replacement battery [same financial unit that budget]
route_time = {i: None for i in N0} #Time of every route a drone follows in a work service [s]


'Function determination and optimisation'

mdl = Model('DRP') #Definition of the model's name (DRP stands for Drone Routing problem)

#Variables definition
x = mdl.addVars(A, vtype = GRB.BINARY) #Edge variable created indexing the set of arcs A. 1 if drone moves from i to j, 0 otherwise.
sigma = mdl.addVars(A, vtype = GRB.BINARY) #Reuse decision variable created indexing the set of arcs A. 1 if drone comes back to the depot and start a new route, 0 otherwise.
y = mdl.addVars(A, vtype = GRB.CONTINUOUS) #Payload weight between locations
t = mdl.addVars(N, vtype = GRB.CONTINUOUS) #Time at which a location is visited
totaltime = mdl.addVar(name='totaltime')
arrival_time = mdl.addVars(N0, vtype = GRB.CONTINUOUS) #Time at which a drone arrives from location i to depot
cost = mdl.addVar(name="cost")
M = mdl.addVar(vtype = GRB.INTEGER, name="M") #Number of drones of the fleet

mdl.modelSense = GRB.MINIMIZE
mdl.setObjective(cost)

#Constraints definition
mdl.addConstr(M>=1)

mdl.addConstrs(quicksum(x[i,j] for j in N if i!=j) == 1 for i in N0) #Locations are only visited once (but the depot).
mdl.addConstrs((quicksum(x[i,j] for j in N if i!=j) - quicksum(x[j,i] for j in N if i!=j)) == 0 for i in N) #Drones arriving at location i also depart from location i.

mdl.addConstrs(quicksum(sigma[i,j] for j in N0 if i!=j) <= x[i,0] for i in N0) #It works with "if i!=j"
mdl.addConstrs(quicksum(sigma[j,i] for j in N0 if i!=j) <= x[0,i] for i in N0) #It works with "if i!=j"
mdl.addConstr((quicksum(x[0,i] for i in N0) - quicksum(sigma[i,j] for i in N0 for j in N0 if i!=j)) <= M) #It works removing the last "s" of add.Constrs

mdl.addConstrs(((quicksum(y[j,i] for j in N if i!=j)) - (quicksum(y[i,j] for j in N if i!=j))) == q[i] for i in N0)
mdl.addConstrs(y[i,j] <= Kub*x[i,j] for i in N for j in N if i!=j)
mdl.addConstrs(y[i,j] <= Q for i in N for j in N if i!=j)

mdl.addConstrs((t[i] - t[j] + tau + d[i,j]/v) <= (Kub*(1-x[i,j])) for i in N for j in N0 if i!=j)
mdl.addConstrs((t[i] - arrival_time[i] + tau + d[i,0]/v) <= (Kub*(1-x[i,0])) for i in N0)
mdl.addConstrs((arrival_time[i] - t[j] + tau + d[0,j]/v) <= (Kub*(1-sigma[i,j])) for i in N0 for j in N0 if i!=j)
mdl.addConstrs(t[i] <= totaltime for i in N)
mdl.addConstr(totaltime <= T)

mdl.addConstr(cost == droneC*M + batteryC*quicksum(sigma[i,j] for i in N0 for j in N0 if i!=j))
mdl.addConstr(cost <= B)


mdl.setParam('Heuristics',0.5)
mdl.Params.TimeLimit = 300  # seconds The optimizer stops if it has been calculated for 30s
mdl.reset() #Apply parameters' modifications
mdl.optimize()

resulting_status = mdl.status
resulting_MIPGap = mdl.MIPGap
resulting_Runtime = mdl.Runtime

if resulting_status == GRB.INFEASIBLE:# or resulting_status == GRB.TIME_LIMIT:
    print('The solution for this DRP is infeasible.')
elif mth.isinf(mdl.ObjVal):
    print('No solution found within the GAP and time limits.')
else:
    cost.ub = cost.x
    mdl.update()
    mdl.setObjective(totaltime)
    
    mdl.Params.TimeLimit = 300  # seconds The optimizer stops if it has been calculated for 30s
    mdl.optimize()
    resulting_status2 = mdl.status
    resulting_MIPGap2 = mdl.MIPGap
    resulting_Runtime2 = mdl.Runtime
    print('An solution for the DRP has been found.')

    
'Additional results'
final_cost = cost.x
final_deliverytime = totaltime.x
number_of_drones = M.x
number_of_routes = quicksum(x[0,i].x for i in N0)
number_of_reusals = quicksum(sigma[i,j].x for i in N0 for j in N0 if i!=j)

for i in N0:
    route_time[i] = 0
    
for i in range(1,n0+1):
    if x[0,i].X > 0.99: #Route departing from depot to location i
        previous_location = 0
        for j in range(1,n0+1):
            if i!=j:
                if sigma[j,i].x > 0.99: #Check if it is a route done with a reused drone
                    previous_location = j
            
        k = i
        l = 0
        f_l_f = False #f_l_f stands for final location found
        while f_l_f == False:
            while x[k,l].x == 0: #Does not leave the while until it finds where the drone goes next
                if k!=(l+1): #To avoid impossible cases (ex. x(1,1))
                    l = l+1
                else:
                    l = l+2
            if l == 0:
                final_location = k
                f_l_f=True
            else: #The drone does not return to the depot yet, so its next step location is searched
                k=l
                l=0
                
        if previous_location == 0:
            route_time[i] = arrival_time[final_location].x
        else:
            route_time[i] = arrival_time[final_location].x-arrival_time[previous_location].x                                 
        
NDA = round((max(route_time[i] for i in N0)/60),2) #Maximum needed drone autonomy [min]

print('The maximum drone autonomy needed is', NDA, 'min.')
                   

'Plotting'

active_arcs = [a for a in A if x[a].x > 0.99]

for i, j in active_arcs:
    plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='g', zorder=0)
plt.plot(xc[0], yc[0], c='r', marker='s')
plt.scatter(xc[1:], yc[1:], c='b')

for n in range(n0+1):
    plt.text(xc[n],yc[n],n, c="r", fontsize = 12)

if xlength <= 100 or ylength <= 100:
    for i, j in active_arcs:
        plt.arrow(xc[i], yc[i], (xc[j]-xc[i])/2, (yc[j]-yc[i])/2, head_width = 0.5)
elif xlength >= 1000 or ylength >= 1000:
    for i, j in active_arcs:
        plt.arrow(xc[i], yc[i], (xc[j]-xc[i])/2, (yc[j]-yc[i])/2, head_width = 9)
else:
    for i, j in active_arcs:
        plt.arrow(xc[i], yc[i], (xc[j]-xc[i])/2, (yc[j]-yc[i])/2, head_width = 5)

plt.xlabel('[m]')
plt.ylabel('[m]')
plt.axis([0,xlength,0,ylength])