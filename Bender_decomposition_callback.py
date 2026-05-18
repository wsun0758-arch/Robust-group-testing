import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from numpy import random
import numpy as np
import timeit,itertools

# Parameters
T = 3  # Number of tests
K = 4  # Number of scenarios
n = 4  # Number of entities
U = 100  # Upper bound on total number of tests

# Subproblem 
def solve_subproblem(x_fixed, T, d):
    # Create a sub
    sub = Model("Subproblem")

    # Decision variables
    z_0_i = sub.addVars(n, lb=0, name="z_0_i")
    z_1_i = sub.addVars(n, lb=0, name="z_1_i")
    z_0_ti = sub.addVars(T, n, lb=0, name="z_0_ti")
    z_1_ti = sub.addVars(T, n, lb=0, name="z_1_ti")

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
 
    # z_{i}^0 ≤ 1
    sub.addConstrs((z_0_i[i] <= 1 for i in range(n)), "C8")
 
    # z_{i}^1 ≤ 1
    sub.addConstrs((z_1_i[i] <= 1 for i in range(n)), "C9")
    
    # z_{ti}^0 ≤ 1
    sub.addConstrs((z_0_ti[t, i] <= 1 for t in range(T) for i in range(n)), "C10")
    
    # z_{ti}^1 ≤ 1
    sub.addConstrs((z_1_ti[t, i] <= 1 for t in range(T) for i in range(n)), "C11")

    # Solve the sub
    sub.optimize()
    sub.params.OutputFlag = 0
    sub.params.LogToConsole = 0
    sub.setParam('DualReductions', 0)
    sub.setParam('InfUnbdInfo', 1)

    # Output the results
    if sub.Status == GRB.OPTIMAL:
        duals={}
        for c in sub.getConstrs():
            duals[c.ConstrName]=c.Pi
            
        for key,value in duals.items():
            match key:
               case "C1","C4","C5": duals[key]=np.reshape(value,(T,n))
               case "C2","C6": duals[key]=np.reshape(value,(T,n,n))
        
        return duals,sub.Objval,sub.Status
    
    else:
        if sub.Status == GRB.INFEASIBLE:
            farkas_dual={}
            for c in sub.getConstrs():
                farkas_dual[c.ConstrName]=c.FarkasDual
                
            for key,value in farkas_dual.items():
                 match key:
                     case "C1","C4","C5": farkas_dual[key]=np.reshape(value,(T,n))
                     case "C2","C6": farkas_dual[key]=np.reshape(value,(T,n,n))
        
        return farkas_dual,-1e+5,sub.Status   


# Bender decompostion
def Bender_decompostion(T,D):
   # Number of items
   n=len(D[0]) 
   
   # Upper bound on the total number of items for an individual test
   U = n  
   
   # Initialize Master Problem
   master = Model("Master Problem")
   x = master.addVars(T, n, vtype=GRB.BINARY, name="x")  # First-stage decision variable
   b = master.addVars(T, vtype=GRB.BINARY, name="b")  # Testing decision variable
   R = master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=U, name="R")  # Auxilary variable
   
   master.params.OutputFlag = 0
   master.params.LogToConsole = 0
   
   # Master problem objective
   master.setObjective(quicksum(b[t] for t in range(T)) - R, GRB.MINIMIZE)
   
   # Capacity constraints
   for t in range(T):
       master.addConstr(quicksum(x[t,i] for i in range(n)) <= U * b[t], name=f"Capacity_{t}")
   
   #Initialize iteration
   global iteration
   iteration=0

   #Initialize upper and lower bounds
   global Upperbound,Lowerbound
   Upperbound=1e+5
   Lowerbound=-1e+5  
   
   global x_opt,b_opt
   x_opt=0
   b_opt=0
   
   def benders_callback(model, where):
       global iteration,Upperbound,Lowerbound,secondstage_optval,x_opt,b_opt
       
       if where == GRB.Callback.MIPSOL:
            
            if Upperbound-Lowerbound>1 and iteration<=100:
                feasibility=1
                
                # 1. Get current best solution and objective value
                x_vals = model.cbGetSolution(model._x)
                b_vals = model.cbGetSolution(model._b)
                R_val = model.cbGetSolution(model._R)
               
                
                # 2. Update iteration
                iteration=iteration+1
                
                secondstage_optval=1e+5
                for d in D:
                    # 3. Solve subproblem with fixed x
                    Sol, Q_value, status = solve_subproblem(x_vals, T, d)
                    
                    if status == GRB.OPTIMAL:
                      secondstage_optval = min(secondstage_optval,Q_value)
       
                      # 4. Check if R is overestimating
                      if R_val > Q_value:
                         # 5. Add lazy constraint (optimality cut)
                         model.cbLazy(quicksum(Sol['C1['+str(t)+','+str(i)+']'] * x[t,i] + Sol['C4['+str(t)+','+str(i)+']'] * x[t,i] for t in range(T) for i in range(n)) +
                            quicksum(Sol['C5['+str(t)+','+str(i)+']']  * quicksum(d[j] * x[t,j] for j in range(n)) for t in range(T) for i in range(n)) +
                            quicksum(Sol['C2['+str(t)+','+str(i)+','+str(j)+']'] * (1 - d[j] * x[t,j]) for t in range(T) for i in range(n) for j in range(n)) +
                            quicksum(Sol['C6['+str(t)+','+str(i)+','+str(j)+']'] * (1 - x[t,j]) for t in range(T) for i in range(n) for j in range(n) if j != i) +
                            quicksum(Sol['C8['+str(i)+']'] + Sol['C9['+str(i)+']'] for i in range(n)) +
                            quicksum(Sol['C10['+str(t)+','+str(i)+']'] + Sol['C11['+str(t)+','+str(i)+']'] for t in range(T) for i in range(n))
                             >=R)
                    else:
                        if status == GRB.INFEASIBLE:
                           feasibility=0
                           # 6. Add lazy constraint (feasibility cut)
                           model.cbLazy(quicksum(Sol['C1['+str(t)+','+str(i)+']'] * x[t,i] + Sol['C4['+str(t)+','+str(i)+']'] * x[t,i] for t in range(T) for i in range(n)) +
                              quicksum(Sol['C5['+str(t)+','+str(i)+']']  * quicksum(d[j] * x[t,j] for j in range(n)) for t in range(T) for i in range(n)) +
                              quicksum(Sol['C2['+str(t)+','+str(i)+','+str(j)+']'] * (1 - d[j] * x[t,j]) for t in range(T) for i in range(n) for j in range(n)) +
                              quicksum(Sol['C6['+str(t)+','+str(i)+','+str(j)+']'] * (1 - x[t,j]) for t in range(T) for i in range(n) for j in range(n) if j != i) +
                              quicksum(Sol['C8['+str(i)+']'] + Sol['C9['+str(i)+']'] for i in range(n)) +
                              quicksum(Sol['C10['+str(t)+','+str(i)+']'] + Sol['C11['+str(t)+','+str(i)+']'] for t in range(T) for i in range(n))
                              >=0)
                       
                 
                if feasibility==1:
                   current_obj=sum(b_vals[t] for t in range(T))-secondstage_optval            
                   if Upperbound>current_obj:
                       Upperbound = current_obj
                       Lowerbound = sum(b_vals[t] for t in range(T))-R_val
                       x_opt = {(t, i): x_vals[t, i].x for t in range(T) for i in range(n)}
                       b_opt = {t: b[t].x for t in range(T)}
            else:
                   model.terminate()
                
   # Store variables for callback
   master._x = x
   master._R = R
   master._b = b
   master.Params.LazyConstraints = 1  # Must be set!
    
   master.optimize(benders_callback)
   
   if Upperbound<1e+5:
       status=2
   else:
       status=0
       
   return x_opt,b_opt,Upperbound,status
 

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
      x_ti = np.reshape(np.array(list(x_ti.values())),(T,n))
      b_t = {t: b[t].x for t in range(T)}
      b_t = np.array(list(b_t.values()))
      for d in D:
         Sol, Q_value, status = solve_subproblem(x_ti, T, d)
         if status == GRB.OPTIMAL:
             # Update the second stage objective function
             secondstage_optval=min(secondstage_optval,Q_value)
      ObjVal=sum(b_t[t] for t in range(T))-secondstage_optval       
      
      return x_ti,b_t,ObjVal,model.status

def enumeration(T,D):
     first_stage_value=1e+5
     for ind, value in enumerate(itertools.product([0, 1], repeat=T*n)):
          x=np.array(value).reshape(T,n)
          b=np.zeros(T)
          second_stage_value=1e+5
          for d in D:     
             Sol,second_stage_Objval,status=solve_subproblem(x, T, d)
             if status==2 and second_stage_value>second_stage_Objval:
               second_stage_value=second_stage_Objval
               xopt=x
          b[np.nonzero(sum(xopt[:,i] for i in range(n)))]=1
          first_stage_Objval=sum(b)-second_stage_value
          if first_stage_value>first_stage_Objval:
              first_stage_value=first_stage_Objval
              x_ti=xopt
              b_i=b 
              
     return x_ti,b_i,first_stage_value
 
iter_max=2#int(np.floor(pow(2,n)/K)))
value_1={i: 1e+5 for i in range(iter_max)}
x_ti_1={i: 0 for i in range(iter_max)}
b_i_1={i: 0 for i in range(iter_max)}
value_2={i: 1e+5 for i in range(iter_max)}
x_ti_2={i: 0 for i in range(iter_max)}
b_i_2={i: 0 for i in range(iter_max)}
value_3={i: 1e+5 for i in range(iter_max)}
x_ti_3={i: 0 for i in range(iter_max)}
b_i_3={i: 0 for i in range(iter_max)}

for i in range(iter_max):
     random.seed(i)
     D = random.randint(0,2,size=(1,n))
     while len(D) < K:
        d = random.randint(0,2,size=n)
        # Check if the number is already in the list
        if len(np.where(~(D-d).any(axis=1))[0])==0:
           D=np.vstack([D,d])

     x_1,b_1,Q_value_1,status_1 = Bender_decompostion(T,D)
     x_2,b_2,Q_value_2,status_2 = One_Stage_MIP(T,D)
     x_3,b_3,Q_value_3 = enumeration(T,D)
     
     if status_1==2 and status_2==2:
        value_1[i] = Q_value_1
        x_ti_1[i] = x_1
        b_i_1[i] = b_1
        value_2[i] = Q_value_2
        x_ti_2[i] = x_2
        b_i_2[i]= b_2
        
     value_3[i] = Q_value_3
     x_ti_3[i] = x_3
     b_i_3[i] = b_3