from gurobipy import Model, GRB, quicksum
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np
import pandas as pd
import timeit
import itertools


# Parameters
T = 10 # Number of tests
K = 5 # Number of scenarios
n = 10  # Number of entities
G = n  # Max number items in each group
r = T # Max times items tested in group tests

# In[1]
def solve_subproblem(x_fixed, T, d):
    # Number of items
    n=len(d) 
        
    # Create a subproblem model
    sub = Model("Maximization Problem")
    
    sub.params.OutputFlag = 0
    sub.params.LogToConsole = 0
    
    # Decision variables
    z_0_i = sub.addVars(n, vtype=GRB.BINARY, name="z_0_i")
    z_1_i = sub.addVars(n, vtype=GRB.BINARY, name="z_1_i")
    z_0_ti = sub.addVars(T, n, vtype=GRB.BINARY, name="z_0_ti")
    z_1_ti = sub.addVars(T, n,vtype=GRB.BINARY, name="z_1_ti")

    # Objective function
    sub.setObjective(
        quicksum(z_0_i[i] + z_1_i[i] for i in range(n)), GRB.MAXIMIZE
    )

    # Constraints
    # z_{ti}^0 ≤ x_{ti}
    sub.addConstrs((z_0_ti[t, i] <= x_fixed[t, i] for t in range(T) for i in range(n)), "C1")

    # z_{ti}^0 ≤ 1 - d_j * x_{tj}, for all t, i, j
    sub.addConstrs(
        (z_0_ti[t, i] <= 1 - d[j] * x_fixed[t, j] for t in range(T) for i in range(n) for j in range(n)), "C2"
    )

    # z_{i}^0 ≤ ∑_{t} z_{ti}^0
    sub.addConstrs((z_0_i[i] <= quicksum(z_0_ti[t, i] for t in range(T)) for i in range(n)), "C3")

    # z_{ti}^1 ≤ x_{ti}
    sub.addConstrs((z_1_ti[t, i] <= x_fixed[t, i] for t in range(T) for i in range(n)), "C4")

    # z_{ti}^1 ≤ ∑_{j} d_j * x_{tj}
    sub.addConstrs(
        (z_1_ti[t, i] <= quicksum(d[j] * x_fixed[t, j] for j in range(n)) for t in range(T) for i in range(n)), "C5"
    )

    # z_{ti}^1 ≤ 1 - x_{tj} + z_{j}^0, for all t, i ≠ j
    sub.addConstrs(
        (z_1_ti[t, i] <= 1 - x_fixed[t, j] + z_0_i[j] for t in range(T) for i in range(n) for j in range(n) if i != j), "C6"
    )

    # z_{i}^1 ≤ ∑_{t} z_{ti}^1
    sub.addConstrs((z_1_i[i] <= quicksum(z_1_ti[t, i] for t in range(T)) for i in range(n)), "C7")
 
    # Solve the sub
    sub.optimize()

    # Output the results
    if sub.status == GRB.OPTIMAL:
        return sub.Objval,sub.status
    else:
        return [1e+5,0]
        
# In[2]
def solve_subproblem_Dual(x_fixed, T,d):
    # number of items
    n=len(d)
    
    sub = Model("Subproblem")
    sub.params.OutputFlag = 0
    sub.params.LogToConsole = 0
    
    # Variables
    v_0 = sub.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="v_0")
    v_1 = sub.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="v_1")
    xi = sub.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="xi")
    eta = sub.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="eta")
    u_0_ti = sub.addVars(T, n, vtype=GRB.CONTINUOUS, lb=0, name="u_0_ti")
    u_1_ti = sub.addVars(T, n, vtype=GRB.CONTINUOUS, lb=0, name="u_1_ti")
    u_2_ti = sub.addVars(T, n, vtype=GRB.CONTINUOUS, lb=0, name="u_2_ti")
    v_0_ti = sub.addVars(T, n, vtype=GRB.CONTINUOUS, lb=0, name="v_0_ti")
    v_1_ti = sub.addVars(T, n, vtype=GRB.CONTINUOUS, lb=0, name="v_1_ti")
    u_0_tij = sub.addVars(T, n, n, vtype=GRB.CONTINUOUS, lb=0, name="u_0_tij")
    u_1_tij = sub.addVars(T, n, n, vtype=GRB.CONTINUOUS, lb=0, name="u_1_tij")

    # Objective function
    sub.setObjective(
        quicksum(u_0_ti[t, i] * x_fixed[t, i] + u_1_ti[t, i] * x_fixed[t, i] for t in range(T) for i in range(n)) +
        quicksum(u_2_ti[t, i] * quicksum(d[j] * x_fixed[t, j] for j in range(n)) for t in range(T) for i in range(n)) +
        quicksum(u_0_tij[t, i, j] * (1 - d[j] * x_fixed[t, j]) for t in range(T) for i in range(n) for j in range(n)) +
        quicksum(u_1_tij[t, i, j] * (1 - x_fixed[t, j]) for t in range(T) for i in range(n) for j in range(n) if j != i) +
        quicksum(v_0[i] + v_1[i] for i in range(n)) +
        quicksum(v_0_ti[t, i] + v_1_ti[t, i] for t in range(T) for i in range(n)),
        GRB.MINIMIZE
    )
    
    # Add subproblem constraints
    sub.addConstrs(
       (u_0_ti[t, i] + quicksum(u_0_tij[t, i, j] for j in range(n)) + v_0_ti[t, i] - xi[i] >= 0 for t in range(T) for i in range(n)),
       "Constraint1"
    )
    
    sub.addConstrs(
    (v_0[i] - quicksum(u_1_tij[t, i, j] for t in range(T) for j in range(n) if j != i) +xi[i] >= 1 for i in range(n)),
    "Constraint2"
    )
    
    sub.addConstrs(
    (u_1_ti[t, i] + u_2_ti[t, i] + quicksum(u_1_tij[t, i, j] for j in range(n) if j != i) + v_1_ti[t, i] - eta[i] >= 0 for t in range(T) for i in range(n)),
    "Constraint3"
    )

    sub.addConstrs(
        (v_1[i] + eta[i] >= 1 for i in range(n)),
        "Constraint4"
    )    

    # Solve subproblem
    sub.setParam(GRB.Param.InfUnbdInfo, 1)
    sub.update()
    sub.optimize()

    n_certified = sub.objval
    
    # Fix decision variables for subproblem
    u_0_ti_fixed=[]
    u_0_tij_fixed=[]
    u_1_ti_fixed=[]
    u_1_tij_fixed=[]
    u_2_ti_fixed=[]
    v_0_fixed=[]
    v_1_fixed=[]
    v_0_ti_fixed=[]
    v_1_ti_fixed=[]
    if sub.status == GRB.OPTIMAL:
            u_0_ti_fixed = {(t, i): u_0_ti[t, i].x for t in range(T) for i in range(n)}
            u_1_ti_fixed = {(t, i): u_1_ti[t, i].x for t in range(T) for i in range(n)}
            u_2_ti_fixed = {(t, i): u_2_ti[t, i].x for t in range(T) for i in range(n)}
            u_0_tij_fixed = {(t, i, j): u_0_tij[t, i, j].x for t in range(T) for i in range(n) for j in range(n)}
            u_1_tij_fixed = {(t, i, j): u_1_tij[t, i, j].x for t in range(T) for i in range(n) for j in range(n)}
            v_0_fixed = {i: v_0[i].x  for i in range(n)}
            v_1_fixed = {i: v_1[i].x  for i in range(n)}
            v_0_ti_fixed = {(t, i): v_0_ti[t, i].x for t in range(T) for i in range(n)}
            v_1_ti_fixed = {(t, i): v_1_ti[t, i].x for t in range(T) for i in range(n)}
    else:
         if sub.status == GRB.UNBOUNDED:
            u_0_ti_fixed = {(t, i): u_0_ti[t, i].UnbdRay for t in range(T) for i in range(n)}
            u_1_ti_fixed = {(t, i): u_1_ti[t, i].UnbdRay for t in range(T) for i in range(n)}
            u_2_ti_fixed = {(t, i): u_2_ti[t, i].UnbdRay for t in range(T) for i in range(n)}
            u_0_tij_fixed = {(t, i, j): u_0_tij[t, i,j].UnbdRay for t in range(T) for i in range(n) for j in range(n)}
            u_1_tij_fixed = {(t, i, j): u_1_tij[t, i, j].UnbdRay for t in range(T) for i in range(n) for j in range(n)}
            v_0_fixed = {i: v_0[i].UnbdRay for i in range(n)}
            v_1_fixed = {i: v_1[i].UnbdRay for i in range(n)}
            v_0_ti_fixed = {(t, i): v_0_ti[t, i].UnbdRay for t in range(T) for i in range(n)}
            v_1_ti_fixed = {(t, i): v_1_ti[t, i].UnbdRay for t in range(T) for i in range(n)}

    return n_certified, (u_0_ti_fixed, u_1_ti_fixed,u_1_tij_fixed,u_2_ti_fixed,u_0_tij_fixed,v_0_ti_fixed,v_1_ti_fixed,v_0_fixed, v_1_fixed),sub.status

# In[3]
def benders_callback(model, where):
     global ub,lb,Upperbound,Lowerbound,x_opt,b_opt,min_result,results,iteration,max_iteration
     if Upperbound-Lowerbound > 1:
       if where == GRB.Callback.MIPSOL and iteration<max_iteration:
              iteration=iteration+1
              
              # 1. Get current best solution and objective value
              x_vals = model.cbGetSolution(model._x)
              b_vals = model.cbGetSolution(model._b)
              R_val = model.cbGetSolution(model._R)
              x_vals = np.array([[x_vals[t, i] for i in range(n)] for t in range(T)])
              b_vals = np.array([b_vals[t] for t in range(T)])
             
              # 2. Parallel execution of subproblems
              def solve_one_scenario(d):
                 return d, solve_subproblem_Dual(x_vals, T, d)
              
              with ThreadPoolExecutor(max_workers=K) as executor:
                 results = list(executor.map(solve_one_scenario, D))
               
              min_result = min(results, key=lambda x: x[1][0])
              d = min_result[0]
              secondstage_optval = min_result[1][0]
              params = min_result[1][1]
              
              u_0_ti_fixed, u_1_ti_fixed,u_1_tij_fixed,u_2_ti_fixed,u_0_tij_fixed,v_0_ti_fixed,v_1_ti_fixed,v_0_fixed, v_1_fixed = params
              
              # 3. Check if R is overestimating
              if R_val > secondstage_optval:
                  # 4. Add lazy constraint (optimality cut)
                  model.cbLazy(quicksum(u_0_ti_fixed[t, i] * model._x[t, i] + u_1_ti_fixed[t, i] * model._x[t, i] for t in range(T) for i in range(n)) +
                  quicksum(u_2_ti_fixed[t, i] * quicksum(d[j] * model._x[t, j] for j in range(n)) for t in range(T) for i in range(n)) +
                  quicksum(u_0_tij_fixed[t, i, j] * (1 - d[j] * model._x[t, j]) for t in range(T) for i in range(n) for j in range(n)) +
                  quicksum(u_1_tij_fixed[t, i, j] * (1 - model._x[t, j]) for t in range(T) for i in range(n) for j in range(n) if j != i) +
                  quicksum(v_0_fixed[i] + v_1_fixed[i] for i in range(n)) +
                  quicksum(v_0_ti_fixed[t, i] + v_1_ti_fixed[t, i] for t in range(T) for i in range(n))
                       >= model._R)
                  
              # 5. Update bounds
              temp=n+sum(b_vals[t] for t in range(T))-secondstage_optval
              if temp<Upperbound:
                  Upperbound=temp
                  x_opt=x_vals
                  b_opt=b_vals
                  Lowerbound=n+sum(b_vals[t] for t in range(T))-R_val
              ub.append(Upperbound)
              lb.append(Lowerbound)
     else:
            model.terminate()

# In[4]
# Bender decompostion
def Bender_decomposition(T,D):
   start = timeit.default_timer()
  
   # Number of items
   n=len(D[0]) 
   
      
   # Initialize Master Problem
   master = Model("Master Problem")
   x = master.addVars(T, n, vtype=GRB.BINARY, name="x")  # First-stage decision variable
   b = master.addVars(T, vtype=GRB.BINARY, name="b")  # Testing decision variable
   R = master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=G, name="R")  # Auxilary variable
   
   master.params.OutputFlag = 0
   master.params.LogToConsole = 0
   master.setParam('TimeLimit', 1800)
   
   # Master problem objective
   master.setObjective(n + quicksum(b[t] for t in range(T)) - R, GRB.MINIMIZE)
   
   # Capacity constraints
   for t in range(T):
       master.addConstr(quicksum(x[t, i] for i in range(n)) <= G * b[t], name=f"Capacity_{t}")
       
   # Test number constraints
   master.addConstrs((quicksum(x[t, i]  for t in range(T)) <= r for i in range(n)), name="Test size") 

   # Second stage objective value
   secondstage_optval=1e+5

   # Iterative Benders decomposition
   Upperbound=1e+5
   Lowerbound=n+T-secondstage_optval
   LB=[]
   UB=[]

   #Initialize iteration
   iteration=0

     
   while Upperbound-Lowerbound > 0 and iteration<100:
       iteration = iteration+1
       # Solve master problem and update the lower bound
       master.optimize()
       if master.status == GRB.OPTIMAL:
            # Fix x variables for subproblem
            x_fixed = {(t, i): x[t, i].x for t in range(T) for i in range(n)}
            x_fixed = np.reshape(np.array(list(x_fixed.values())),(T,n))
            # Output b variables for subproblem
            b_t = {t: b[t].x for t in range(T)}
            b_t = np.array(list(b_t.values()))
            # Update the lower bound
            Lowerbound = master.ObjVal
            LB.append(Lowerbound)
       else:
            break
     
       # 1. Second stage objective value
       secondstage_optval=1e+5

       # 2. Parallel execution of subproblems
       def solve_one_scenario(d):
          return d, solve_subproblem_Dual(x_fixed, T, d)
       
       with ThreadPoolExecutor(max_workers=K) as executor:
          results = list(executor.map(solve_one_scenario, D))

       min_result = min(results, key=lambda x: x[1][0])
       d = min_result[0]
       secondstage_optval = min_result[1][0]
       params = min_result[1][1]
       
       u_0_ti_fixed, u_1_ti_fixed,u_1_tij_fixed,u_2_ti_fixed,u_0_tij_fixed,v_0_ti_fixed,v_1_ti_fixed,v_0_fixed, v_1_fixed = params
      
       # 3. Add optimality cut
       master.addConstr(quicksum(u_0_ti_fixed[t, i] * x[t, i] + u_1_ti_fixed[t, i] * x[t, i] for t in range(T) for i in range(n)) +
                   quicksum(u_2_ti_fixed[t, i] * quicksum(d[j] * x[t, j] for j in range(n)) for t in range(T) for i in range(n)) +
                   quicksum(u_0_tij_fixed[t, i, j] * (1 - d[j] * x[t, j]) for t in range(T) for i in range(n) for j in range(n)) +
                   quicksum(u_1_tij_fixed[t, i, j] * (1 - x[t, j]) for t in range(T) for i in range(n) for j in range(n) if j != i) +
                   quicksum(v_0_fixed[i] + v_1_fixed[i] for i in range(n)) +
                   quicksum(v_0_ti_fixed[t, i] + v_1_ti_fixed[t, i] for t in range(T) for i in range(n))
                    >=R, name="Optimality Cut")

       master.update()           
       
       # 4. Update the upper bound
       temp=n+sum(b_t[t] for t in range(T))-secondstage_optval
       if temp<Upperbound:
           Upperbound=temp
           x_fixed_0=x_fixed
           b_t_0=b_t
       
       UB.append(Upperbound)
       
   stop = timeit.default_timer()
   Time = stop - start
 
   return x_fixed_0,b_t_0,Upperbound,master.status,UB,LB,Time


iter_max=1#int(np.floor(pow(2,n)/K))
value={i: 1e+5 for i in range(iter_max)}
x_ti={i: 0 for i in range(iter_max)}
b_i={i: 0 for i in range(iter_max)}
status={i: 0 for i in range(iter_max)}
Time={i: 0 for i in range(iter_max)}
gap={i: 0 for i in range(iter_max)}
ub_BD={i: 0 for i in range(iter_max)}
lb_BD={i: 0 for i in range(iter_max)}

m=int(n/2)

for i in range(iter_max):
    random.seed(i)
    index = random.sample(range(n), m)
    D=np.zeros(n)
    D[list(index)]+=1
    D=np.reshape(D, (1,n))  
    while len(D) < K:
       index = random.sample(range(n), m)
       d=np.zeros(n)
       d[list(index)]+=1
       d=np.reshape(d, (1,n))  
       # Check if the number is already in the list
       if len(np.where(~(D-d).any(axis=1))[0])==0:
           D=np.vstack([D,d])  '
           

    x_ti[i],b_i[i],ub_BD[i],lb_BD[i],value[i],gap[i],status[i],Time[i] = Bender_decomposition(T, D)
    








