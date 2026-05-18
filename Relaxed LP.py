from gurobipy import Model, GRB, quicksum
from numpy import random
import gurobipy as gp
import numpy as np

# Parameters
K = 4  # Number of scenarios
T = 2  # Number of tests
n = 5  # Number of entities
U = 100  # Upper bound on total number of tests

# Subproblem (Prime)
def solve_subproblem_Prime(x_fixed, d):
    # Create a sub
    sub = Model("Maximization Problem")

    # Decision variables
    z_0_i = sub.addVars(n, lb=0, ub=1, name="z_0_i")
    z_1_i = sub.addVars(n, lb=0, ub=1, name="z_1_i")
    z_0_ti = sub.addVars(T, n, lb=0, ub=1, name="z_0_ti")
    z_1_ti = sub.addVars(T, n, lb=0, ub=1, name="z_1_ti")

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
        z_0_i_fixed={i:z_0_i[i].x for i in range(n)}
        z_1_i_fixed={i:z_1_i[i].x for i in range(n)}
        z_0_ti_fixed={(t,i):z_0_ti[t, i].x for i in range(n) for t in range(T)}
        z_1_ti_fixed={(t,i):z_1_ti[t, i].x for i in range(n) for t in range(T)}
  
    return z_0_i_fixed,z_1_i_fixed,z_0_ti_fixed,z_1_ti_fixed,sub.Objval,sub.status
        
# Subproblem (Dual)
def solve_subproblem_Dual(x_fixed, d):
    print(x_fixed)
    sub = Model("Subproblem")
    sub.setParam("OutputFlag", 0)  # Suppress output

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
    (v_0[i] - quicksum(u_1_tij[t, j, i] for t in range(T) for j in range(n) if j != i) +xi[i] >= 1 for i in range(n)),
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
    sub.optimize()
    
    # Fix decision variables for subproblem
    u_0_ti_fixed=[]
    u_1_ti_fixed=[]
    u_2_ti_fixed=[]
    u_0_tij_fixed=[]
    if sub.status == GRB.OPTIMAL:
            u_0_ti_fixed = {(t, i): u_0_ti[t, i].x for t in range(T) for i in range(n)}
            u_1_ti_fixed = {(t, i): u_1_ti[t, i].x for t in range(T) for i in range(n)}
            u_2_ti_fixed = {(t, i): u_2_ti[t, i].x for t in range(T) for i in range(n)}
            u_0_tij_fixed = {(t, i, j): u_0_tij[t, i, j].x for t in range(T) for i in range(n) for j in range(n)}
    
    return u_0_ti_fixed, u_1_ti_fixed,u_2_ti_fixed,u_0_tij_fixed,sub.Objval,sub.status   

def Nam_subproblem_Dual(x_fixed, d):
    model = gp.Model("SecondStageDual")
    
    # Variables
    lambda_var = model.addVars(T, n, name="lambda", lb=0)
    gamma = model.addVars(T, n, n, name="gamma", lb=0)
    xi = model.addVars(n, name="xi", lb=0)
    eta = model.addVars(T, n, name="eta", lb=0)
    delta = model.addVars(T, n, name="delta", lb=0)
    kappa = model.addVars(T, n, n, name="kappa", lb=0)
    zeta = model.addVars(n, name="zeta", lb=0)
    mu0 = model.addVars(n, name="mu0", lb=0)
    mu1 = model.addVars(n, name="mu1", lb=0)
    
    # Objective (15a)
    obj = gp.quicksum(
        x_fixed[t,i] * (lambda_var[t, i] + eta[t, i] + gp.quicksum(d[i1] * x_fixed[t,i1] * delta[t, i] for i1 in range(n)))
        for t in range(T) for i in range(n)
    ) + gp.quicksum(
        (1 - d[i1] * x_fixed[t,i1]) * gamma[t, i, i1]
        for t in range(T) for i in range(n) for i1 in range(n)
    ) + gp.quicksum(
        (1 - x_fixed[t,j]) * kappa[t, i, j]
        for t in range(T) for i in range(n) for j in range(n) if j != i
    ) + gp.quicksum(mu0[i] + mu1[i] for i in range(n))
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints (15b)
    for i in range(n):
        model.addConstr(
            1 <= mu0[i] - gp.quicksum(kappa[t, i, j] for t in range(T) for j in range(n) if j != i) + xi[i],
            name=f"constr_15b_{i}"
        )
    
    # Constraints (15c)
    for i in range(n):
        model.addConstr(1 <= mu1[i] + zeta[i], name=f"constr_15c_{i}")
    
    # Constraints (15d)
    for t in range(T):
        for i in range(n):
            model.addConstr(lambda_var[t, i] + gp.quicksum(gamma[t, i, i1] for i1 in range(n)) - xi[i] >= 0,
                            name=f"constr_15d_{t}_{i}")
    
    # Constraints (15e)
    for t in range(T):
        for i in range(n):
            model.addConstr(
                eta[t, i] + delta[t, i] + gp.quicksum(kappa[t, i, j] for j in range(n) if j != i) - zeta[i] >= 0,
                name=f"constr_15e_{t}_{i}"
            )
    
    # Optimize
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
       return model.Objval,model.status   

iter_max=2
Q_value_diff_1={i: 0 for i in range(iter_max)}
Q_value_diff_2={i: 0 for i in range(iter_max)}
Q_value_Prime={i: 0 for i in range(iter_max)}
Q_value_Dual={i: 0 for i in range(iter_max)}
Q_value_Nam={i: 0 for i in range(iter_max)}
D={i: 0 for i in range(iter_max)}
x_fixed={i: 0 for i in range(iter_max)}

for i in range(iter_max):
     random.seed(i)
     D[i] = random.randint(0,2,size=(1,n))
     while len(D[i]) < K:
        d = random.randint(0,2,size=n)
        # Check if the number is already in the list
        if len(np.where(~(D[i]-d).any(axis=1))[0])==0:
           D[i]=np.vstack([D[i],d])
           
     x_fixed[i]=random.randint(0,2, size=(T, n))
     
     z_0_i_fixed,z_1_i_fixed,z_0_ti_fixed,z_1_ti_fixed,Q_value_Prime[i],status_Prime=solve_subproblem_Prime(x_fixed[i], D[i])
     u_0_ti_fixed, u_1_ti_fixed,u_2_ti_fixed,u_0_tij_fixed, Q_value_Dual[i], status_Dual = solve_subproblem_Dual(x_fixed[i], D[i])
     Q_value_Nam[i], status_Nam = Nam_subproblem_Dual(x_fixed[i], D[i])
    
     if status_Nam==2 and status_Prime==2:
        Q_value_diff_1[i] = Q_value_Prime[i] - Q_value_Nam[i]
     if status_Dual==2 and status_Prime==2:
        Q_value_diff_2[i] = Q_value_Prime[i] - Q_value_Dual[i]