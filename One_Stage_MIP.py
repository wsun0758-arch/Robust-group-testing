from gurobipy import Model, GRB, quicksum
from numpy import random
import numpy as np

# Parameters
K = 3  # Number of scenarios
T = 2  # Number of tests
n = 5  # Number of entities
U = 100  # Upper bound on total number of tests
iter_max=4#int(np.floor(pow(2,n)/K)))

# Subproblem 
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

# One_Stage_MIP
def One_Stage_MIP(T,D):
      
   # Number of items
   n=len(D[0]) 

   # Upper bound on total number of items for an individual test
   U = n  

   # Number of scenarios
   K=len(D)
   
   # Create model
   model = Model("One_Stage_MIP")

   model.params.OutputFlag = 0
   model.params.LogToConsole = 0
   
   # Decision variables
   x = model.addVars(T, n, vtype=GRB.BINARY, name="x")
   b = model.addVars(T, vtype=GRB.BINARY, name="b")
   R = model.addVar(vtype=GRB.CONTINUOUS, name="R")
   z_0_ti = model.addVars(T, n, K, vtype=GRB.BINARY, name="z_0_ti")
   z_1_ti = model.addVars(T, n, K, vtype=GRB.BINARY, name="z_1_ti")
   z_0_i = model.addVars(n, K, vtype=GRB.BINARY, name="z_0_i")
   z_1_i = model.addVars(n, K, vtype=GRB.BINARY, name="z_1_i")

   # Objective function
   model.setObjective(quicksum(b[t] for t in range(T)) - R, GRB.MINIMIZE)
          
   # Constraints
   # 1. Capacity constraint
   model.addConstrs((quicksum(x[t, i] for i in range(n)) <= U * b[t] for t in range(T)), "Capacity")
          
   # 2. R upper bound
   model.addConstrs((R <= quicksum(z_0_i[i, k] + z_1_i[i, k] for i in range(n)) for k in range(K)), "R_Upper_Bound")

   # 3. z_0_ti constraints
   model.addConstrs((z_0_ti[t, i, k] <= x[t, i] for t in range(T) for i in range(n) for k in range(K)), "z_0_ti_L1")

   model.addConstrs((z_0_ti[t, i, k] <= 1 - D[k,j] * x[t, j] 
                         for t in range(T) for i in range(n) for k in range(K) for j in range(n)), "z_0_ti_L2")
   model.addConstrs((z_0_i[i, k] <= quicksum(z_0_ti[t, i, k] for t in range(T)) 
                         for i in range(n) for k in range(K)), "z_0_i_Sum")
       
   # 4. z_1_ti constraints
   model.addConstrs((z_1_ti[t, i, k] <= x[t, i] for t in range(T) for i in range(n) for k in range(K)), "z_1_ti_L1")
   model.addConstrs((z_1_ti[t, i, k] <= quicksum(D[k,j] * x[t, j] for j in range(n)) 
                         for t in range(T) for i in range(n) for k in range(K)), "z_1_ti_L2")
   model.addConstrs((z_1_ti[t, i, k] <= 1 - x[t, j] + z_0_i[j, k] 
                         for t in range(T) for i in range(n) for j in range(n) if j != i for k in range(K)), "z_1_ti_L3")
   model.addConstrs((z_1_i[i, k] <= quicksum(z_1_ti[t, i, k] for t in range(T)) 
                         for i in range(n) for k in range(K)), "z_1_i_Sum")
       
   # Solve the model
   model.optimize()
       
   x_ti = {(t, i): 0 for t in range(T) for i in range(n)}
   b_t = {t: 1 for t in range(T)}
   # Second stage objective value
   secondstage_optval=1e+5
   
   # Display the results
   if model.status == GRB.OPTIMAL:
      x_ti = {(t, i): x[t, i].x for t in range(T) for i in range(n)}
      x_ti=np.reshape(np.array(list(x_ti.values())),(T,n))
      b_t = {t: b[t].x for t in range(T)}
      b_t=np.array(list(b_t.values()))
      for d in D:
         Q_value, status = solve_subproblem(x_ti,T, d)
         if status == GRB.OPTIMAL:
             # Update the second stage objective function
             secondstage_optval=min(secondstage_optval,Q_value)
             
      ObjVal=sum(b_t[t] for t in range(T))-secondstage_optval       
      
        
   return x_ti,b_t,ObjVal,model.status


value_1={i: 1e+5 for i in range(iter_max)}
x_ti_1={i: 0 for i in range(iter_max)}
b_i_1={i: 0 for i in range(iter_max)}

for i in range(iter_max):
     random.seed(i)
     D = random.randint(0,2,size=(1,n))
         
     while len(D) < K:
        d = random.randint(0,2,size=n)
        
        # Check if the number is already in the list
        if len(np.where(~(D-d).any(axis=1))[0])==0:
           D=np.vstack([D,d])
   
        x_1,b_1,Q_value_1,status_1 = One_Stage_MIP(T,D)
     
        if status_1==2:
            value_1[i] = Q_value_1
            x_ti_1[i] = x_1
            b_i_1[i]= b_1