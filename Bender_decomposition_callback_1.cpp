#include "gurobi_c++.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Tensor3 = std::vector<Matrix>;

// ---------------------------------------------------------------------------
// Parameters (module-level globals in the Python source)
// ---------------------------------------------------------------------------
constexpr int T = 10;   // Number of tests
constexpr int K = 5;    // Number of scenarios
constexpr int n = 10;   // Number of entities
constexpr int G = n;    // Max number items in each group
constexpr int r = T;    // Max times items tested in group tests
constexpr int MAX_ITERATION = 100;

// ---------------------------------------------------------------------------
// Dual solution of the subproblem, used to build Benders optimality cuts.
// ---------------------------------------------------------------------------
struct DualCutParams {
    Matrix u_0_ti;    // T x n
    Matrix u_1_ti;    // T x n
    Tensor3 u_1_tij;  // T x n x n
    Matrix u_2_ti;    // T x n
    Tensor3 u_0_tij;  // T x n x n
    Matrix v_0_ti;    // T x n
    Matrix v_1_ti;    // T x n
    Vector v_0;       // n
    Vector v_1;       // n
};

// ---------------------------------------------------------------------------
// In[1]: primal subproblem (maximization). Not called by the main pipeline,
// but kept for parity with the Python source.
// ---------------------------------------------------------------------------
std::pair<double, int> solve_subproblem(const Matrix& x_fixed, int T_local, const Vector& d) {
    int n_local = (int)d.size();

    GRBEnv env = GRBEnv();
    GRBModel sub(env);
    sub.set(GRB_IntParam_OutputFlag, 0);
    sub.set(GRB_IntParam_LogToConsole, 0);

    std::vector<GRBVar> z_0_i(n_local), z_1_i(n_local);
    for (int i = 0; i < n_local; i++) {
        z_0_i[i] = sub.addVar(0, 1, 0, GRB_BINARY, "z_0_i_" + std::to_string(i));
        z_1_i[i] = sub.addVar(0, 1, 0, GRB_BINARY, "z_1_i_" + std::to_string(i));
    }

    std::vector<std::vector<GRBVar>> z_0_ti(T_local, std::vector<GRBVar>(n_local));
    std::vector<std::vector<GRBVar>> z_1_ti(T_local, std::vector<GRBVar>(n_local));
    for (int t = 0; t < T_local; t++) {
        for (int i = 0; i < n_local; i++) {
            z_0_ti[t][i] = sub.addVar(0, 1, 0, GRB_BINARY, "z_0_ti_" + std::to_string(t) + "_" + std::to_string(i));
            z_1_ti[t][i] = sub.addVar(0, 1, 0, GRB_BINARY, "z_1_ti_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    GRBLinExpr obj = 0;
    for (int i = 0; i < n_local; i++) obj += z_0_i[i] + z_1_i[i];
    sub.setObjective(obj, GRB_MAXIMIZE);

    // C1: z_0_ti <= x_fixed
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            sub.addConstr(z_0_ti[t][i] <= x_fixed[t][i], "C1_" + std::to_string(t) + "_" + std::to_string(i));

    // C2: z_0_ti <= 1 - d_j * x_fixed[t,j]
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int j = 0; j < n_local; j++)
                sub.addConstr(z_0_ti[t][i] <= 1 - d[j] * x_fixed[t][j],
                              "C2_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));

    // C3: z_0_i <= sum_t z_0_ti
    for (int i = 0; i < n_local; i++) {
        GRBLinExpr s = 0;
        for (int t = 0; t < T_local; t++) s += z_0_ti[t][i];
        sub.addConstr(z_0_i[i] <= s, "C3_" + std::to_string(i));
    }

    // C4: z_1_ti <= x_fixed
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            sub.addConstr(z_1_ti[t][i] <= x_fixed[t][i], "C4_" + std::to_string(t) + "_" + std::to_string(i));

    // C5: z_1_ti <= sum_j d_j * x_fixed[t,j]
    for (int t = 0; t < T_local; t++) {
        for (int i = 0; i < n_local; i++) {
            double s = 0;
            for (int j = 0; j < n_local; j++) s += d[j] * x_fixed[t][j];
            sub.addConstr(z_1_ti[t][i] <= s, "C5_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    // C6: z_1_ti <= 1 - x_fixed[t,j] + z_0_i[j], for i != j
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int j = 0; j < n_local; j++)
                if (i != j)
                    sub.addConstr(z_1_ti[t][i] <= 1 - x_fixed[t][j] + z_0_i[j],
                                  "C6_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));

    // C7: z_1_i <= sum_t z_1_ti
    for (int i = 0; i < n_local; i++) {
        GRBLinExpr s = 0;
        for (int t = 0; t < T_local; t++) s += z_1_ti[t][i];
        sub.addConstr(z_1_i[i] <= s, "C7_" + std::to_string(i));
    }

    sub.optimize();

    if (sub.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
        return {sub.get(GRB_DoubleAttr_ObjVal), GRB_OPTIMAL};
    } else {
        return {1e5, 0};
    }
}

// ---------------------------------------------------------------------------
// In[2]: dual subproblem, used to generate Benders optimality cuts.
// ---------------------------------------------------------------------------
std::tuple<double, DualCutParams, int> solve_subproblem_Dual(const Matrix& x_fixed, int T_local, const Vector& d) {
    int n_local = (int)d.size();

    GRBEnv env = GRBEnv();
    GRBModel sub(env);
    sub.set(GRB_IntParam_OutputFlag, 0);
    sub.set(GRB_IntParam_LogToConsole, 0);

    std::vector<GRBVar> v_0(n_local), v_1(n_local), xi(n_local), eta(n_local);
    for (int i = 0; i < n_local; i++) {
        v_0[i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "v_0_" + std::to_string(i));
        v_1[i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "v_1_" + std::to_string(i));
        xi[i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "xi_" + std::to_string(i));
        eta[i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "eta_" + std::to_string(i));
    }

    std::vector<std::vector<GRBVar>> u_0_ti(T_local, std::vector<GRBVar>(n_local));
    std::vector<std::vector<GRBVar>> u_1_ti(T_local, std::vector<GRBVar>(n_local));
    std::vector<std::vector<GRBVar>> u_2_ti(T_local, std::vector<GRBVar>(n_local));
    std::vector<std::vector<GRBVar>> v_0_ti(T_local, std::vector<GRBVar>(n_local));
    std::vector<std::vector<GRBVar>> v_1_ti(T_local, std::vector<GRBVar>(n_local));
    for (int t = 0; t < T_local; t++) {
        for (int i = 0; i < n_local; i++) {
            u_0_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "u_0_ti_" + std::to_string(t) + "_" + std::to_string(i));
            u_1_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "u_1_ti_" + std::to_string(t) + "_" + std::to_string(i));
            u_2_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "u_2_ti_" + std::to_string(t) + "_" + std::to_string(i));
            v_0_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "v_0_ti_" + std::to_string(t) + "_" + std::to_string(i));
            v_1_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "v_1_ti_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    std::vector<std::vector<std::vector<GRBVar>>> u_0_tij(
        T_local, std::vector<std::vector<GRBVar>>(n_local, std::vector<GRBVar>(n_local)));
    std::vector<std::vector<std::vector<GRBVar>>> u_1_tij(
        T_local, std::vector<std::vector<GRBVar>>(n_local, std::vector<GRBVar>(n_local)));
    for (int t = 0; t < T_local; t++) {
        for (int i = 0; i < n_local; i++) {
            for (int j = 0; j < n_local; j++) {
                u_0_tij[t][i][j] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS,
                                               "u_0_tij_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));
                u_1_tij[t][i][j] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS,
                                               "u_1_tij_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
    }

    // Objective function
    GRBLinExpr obj = 0;
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            obj += u_0_ti[t][i] * x_fixed[t][i] + u_1_ti[t][i] * x_fixed[t][i];

    for (int t = 0; t < T_local; t++) {
        for (int i = 0; i < n_local; i++) {
            double s = 0;
            for (int j = 0; j < n_local; j++) s += d[j] * x_fixed[t][j];
            obj += u_2_ti[t][i] * s;
        }
    }

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int j = 0; j < n_local; j++)
                obj += u_0_tij[t][i][j] * (1 - d[j] * x_fixed[t][j]);

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int j = 0; j < n_local; j++)
                if (j != i)
                    obj += u_1_tij[t][i][j] * (1 - x_fixed[t][j]);

    for (int i = 0; i < n_local; i++) obj += v_0[i] + v_1[i];

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            obj += v_0_ti[t][i] + v_1_ti[t][i];

    sub.setObjective(obj, GRB_MINIMIZE);

    // Constraint1 (the stray "0" in the Python source is dropped - see file header note)
    for (int t = 0; t < T_local; t++) {
        for (int i = 0; i < n_local; i++) {
            GRBLinExpr expr = u_0_ti[t][i] + v_0_ti[t][i] - xi[i];
            for (int j = 0; j < n_local; j++) expr += u_0_tij[t][i][j];
            sub.addConstr(expr >= 0, "Constraint1_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    // Constraint2
    for (int i = 0; i < n_local; i++) {
        GRBLinExpr expr = v_0[i] + xi[i];
        for (int t = 0; t < T_local; t++)
            for (int j = 0; j < n_local; j++)
                if (j != i) expr -= u_1_tij[t][i][j];
        sub.addConstr(expr >= 1, "Constraint2_" + std::to_string(i));
    }

    // Constraint3
    for (int t = 0; t < T_local; t++) {
        for (int i = 0; i < n_local; i++) {
            GRBLinExpr expr = u_1_ti[t][i] + u_2_ti[t][i] + v_1_ti[t][i] - eta[i];
            for (int j = 0; j < n_local; j++)
                if (j != i) expr += u_1_tij[t][i][j];
            sub.addConstr(expr >= 0, "Constraint3_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    // Constraint4
    for (int i = 0; i < n_local; i++)
        sub.addConstr(v_1[i] + eta[i] >= 1, "Constraint4_" + std::to_string(i));

    sub.set(GRB_IntParam_InfUnbdInfo, 1);
    sub.update();
    sub.optimize();

    // Mirrors Python's unconditional `n_certified = sub.objval` read right after optimize().
    double n_certified = sub.get(GRB_DoubleAttr_ObjVal);
    int status = sub.get(GRB_IntAttr_Status);

    DualCutParams p;
    p.u_0_ti = Matrix(T_local, Vector(n_local, 0.0));
    p.u_1_ti = Matrix(T_local, Vector(n_local, 0.0));
    p.u_2_ti = Matrix(T_local, Vector(n_local, 0.0));
    p.v_0_ti = Matrix(T_local, Vector(n_local, 0.0));
    p.v_1_ti = Matrix(T_local, Vector(n_local, 0.0));
    p.u_0_tij = Tensor3(T_local, Matrix(n_local, Vector(n_local, 0.0)));
    p.u_1_tij = Tensor3(T_local, Matrix(n_local, Vector(n_local, 0.0)));
    p.v_0 = Vector(n_local, 0.0);
    p.v_1 = Vector(n_local, 0.0);

    if (status == GRB_OPTIMAL) {
        for (int t = 0; t < T_local; t++) {
            for (int i = 0; i < n_local; i++) {
                p.u_0_ti[t][i] = u_0_ti[t][i].get(GRB_DoubleAttr_X);
                p.u_1_ti[t][i] = u_1_ti[t][i].get(GRB_DoubleAttr_X);
                p.u_2_ti[t][i] = u_2_ti[t][i].get(GRB_DoubleAttr_X);
                p.v_0_ti[t][i] = v_0_ti[t][i].get(GRB_DoubleAttr_X);
                p.v_1_ti[t][i] = v_1_ti[t][i].get(GRB_DoubleAttr_X);
                for (int j = 0; j < n_local; j++) {
                    p.u_0_tij[t][i][j] = u_0_tij[t][i][j].get(GRB_DoubleAttr_X);
                    p.u_1_tij[t][i][j] = u_1_tij[t][i][j].get(GRB_DoubleAttr_X);
                }
            }
        }
        for (int i = 0; i < n_local; i++) {
            p.v_0[i] = v_0[i].get(GRB_DoubleAttr_X);
            p.v_1[i] = v_1[i].get(GRB_DoubleAttr_X);
        }
    } else if (status == GRB_UNBOUNDED) {
        for (int t = 0; t < T_local; t++) {
            for (int i = 0; i < n_local; i++) {
                p.u_0_ti[t][i] = u_0_ti[t][i].get(GRB_DoubleAttr_UnbdRay);
                p.u_1_ti[t][i] = u_1_ti[t][i].get(GRB_DoubleAttr_UnbdRay);
                p.u_2_ti[t][i] = u_2_ti[t][i].get(GRB_DoubleAttr_UnbdRay);
                p.v_0_ti[t][i] = v_0_ti[t][i].get(GRB_DoubleAttr_UnbdRay);
                p.v_1_ti[t][i] = v_1_ti[t][i].get(GRB_DoubleAttr_UnbdRay);
                for (int j = 0; j < n_local; j++) {
                    p.u_0_tij[t][i][j] = u_0_tij[t][i][j].get(GRB_DoubleAttr_UnbdRay);
                    p.u_1_tij[t][i][j] = u_1_tij[t][i][j].get(GRB_DoubleAttr_UnbdRay);
                }
            }
        }
        for (int i = 0; i < n_local; i++) {
            p.v_0[i] = v_0[i].get(GRB_DoubleAttr_UnbdRay);
            p.v_1[i] = v_1[i].get(GRB_DoubleAttr_UnbdRay);
        }
    }

    return {n_certified, p, status};
}

// Solve all K scenarios for a fixed x, in parallel, and return the argmin by objective value.
static std::tuple<Vector, double, DualCutParams> solve_all_scenarios_min(const Matrix& x_fixed, int T_local,
                                                                          const std::vector<Vector>& D) {
    std::vector<std::future<std::tuple<Vector, double, DualCutParams, int>>> futures;
    futures.reserve(D.size());
    for (const auto& d : D) {
        futures.push_back(std::async(std::launch::async, [x_fixed, T_local, d]() {
            auto result = solve_subproblem_Dual(x_fixed, T_local, d);
            return std::make_tuple(d, std::get<0>(result), std::get<1>(result), std::get<2>(result));
        }));
    }

    std::vector<std::tuple<Vector, double, DualCutParams, int>> results;
    results.reserve(futures.size());
    for (auto& f : futures) results.push_back(f.get());

    auto minIt = std::min_element(results.begin(), results.end(),
                                   [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });

    return {std::get<0>(*minIt), std::get<1>(*minIt), std::get<2>(*minIt)};
}

// Builds the left-hand side of the Benders optimality cut  cut >= R  over variables `xvar`.
// The caller decides how to add it: addLazy() inside a callback, addConstr() outside one.
static GRBLinExpr buildOptimalityCut(std::vector<std::vector<GRBVar>>& xvar, const Vector& d, const DualCutParams& p) {
    GRBLinExpr cut = 0;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            cut += p.u_0_ti[t][i] * xvar[t][i] + p.u_1_ti[t][i] * xvar[t][i];

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            GRBLinExpr s = 0;
            for (int j = 0; j < n; j++) s += d[j] * xvar[t][j];
            cut += p.u_2_ti[t][i] * s;
        }
    }

    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                cut += p.u_0_tij[t][i][j] * (1 - d[j] * xvar[t][j]);

    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (j != i) cut += p.u_1_tij[t][i][j] * (1 - xvar[t][j]);

    for (int i = 0; i < n; i++) cut += p.v_0[i] + p.v_1[i];

    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            cut += p.v_0_ti[t][i] + p.v_1_ti[t][i];

    return cut;
}

// ---------------------------------------------------------------------------
// In[3]: Benders lazy-constraint callback.
// Encapsulates the state that Python shares via `global` between
// benders_callback() and Bender_decomposition().
// ---------------------------------------------------------------------------
class BendersCallback : public GRBCallback {
public:
    BendersCallback(std::vector<std::vector<GRBVar>>& x_, std::vector<GRBVar>& b_, GRBVar R_,
                    const std::vector<Vector>& D_, GRBModel* model_)
        : x(x_), b(b_), R(R_), D(D_), model(model_), Upperbound(1e5), Lowerbound(-1e5), iteration(0) {
        x_opt = Matrix(T, Vector(n, 0.0));
        b_opt = Vector(T, 0.0);
    }

    const Matrix& getXOpt() const { return x_opt; }
    const Vector& getBOpt() const { return b_opt; }
    const Vector& getUB() const { return ub; }
    const Vector& getLB() const { return lb; }
    double getUpperbound() const { return Upperbound; }

protected:
    void callback() override {
        try {
            if (Upperbound - Lowerbound > 1) {
                if (where == GRB_CB_MIPSOL && iteration < MAX_ITERATION) {
                    iteration++;

                    // 1. Get current best solution and objective value
                    Matrix x_vals(T, Vector(n));
                    for (int t = 0; t < T; t++)
                        for (int i = 0; i < n; i++)
                            x_vals[t][i] = getSolution(x[t][i]);

                    Vector b_vals(T);
                    for (int t = 0; t < T; t++) b_vals[t] = getSolution(b[t]);

                    double R_val = getSolution(R);

                    // 2. Parallel execution of subproblems
                    auto [d, secondstage_optval, p] = solve_all_scenarios_min(x_vals, T, D);

                    // 3. Check if R is overestimating
                    if (R_val > secondstage_optval) {
                        // 4. Add lazy constraint (optimality cut)
                        GRBLinExpr cut = buildOptimalityCut(x, d, p);
                        addLazy(cut >= R);
                    }

                    // 5. Update bounds
                    double bsum = 0;
                    for (int t = 0; t < T; t++) bsum += b_vals[t];
                    double temp = n + bsum - secondstage_optval;
                    if (temp < Upperbound) {
                        Upperbound = temp;
                        x_opt = x_vals;
                        b_opt = b_vals;
                        Lowerbound = n + bsum - R_val;
                    }
                    ub.push_back(Upperbound);
                    lb.push_back(Lowerbound);
                }
            } else {
                model->terminate();
            }
        } catch (GRBException& e) {
            std::cerr << "Callback error " << e.getErrorCode() << ": " << e.getMessage() << std::endl;
        } catch (...) {
            std::cerr << "Unknown callback error" << std::endl;
        }
    }

private:
    // Note: addLazy() requires a GRBLinExpr; R (a GRBVar) converts implicitly where needed.
    std::vector<std::vector<GRBVar>>& x;
    std::vector<GRBVar>& b;
    GRBVar R;
    const std::vector<Vector>& D;
    GRBModel* model;

    double Upperbound, Lowerbound;
    int iteration;
    Matrix x_opt;
    Vector b_opt;
    Vector ub, lb;
};

// ---------------------------------------------------------------------------
// In[3]/In[4]: master problem solved with the Benders lazy-constraint callback.
// ---------------------------------------------------------------------------
struct BDResult {
    Matrix x_opt;
    Vector b_opt;
    Vector ub;
    Vector lb;
    double Upperbound;
    double gap;
    int status;
    double runtime;
};

BDResult Bender_decomposition(int T_local, const std::vector<Vector>& D) {
    GRBEnv env = GRBEnv();
    GRBModel master(env);
    master.set(GRB_IntParam_OutputFlag, 0);
    master.set(GRB_IntParam_LogToConsole, 0);
    master.set(GRB_DoubleParam_TimeLimit, 1800);

    std::vector<std::vector<GRBVar>> x(T_local, std::vector<GRBVar>(n));
    std::vector<GRBVar> b(T_local);
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            x[t][i] = master.addVar(0, 1, 0, GRB_BINARY, "x_" + std::to_string(t) + "_" + std::to_string(i));
    for (int t = 0; t < T_local; t++) b[t] = master.addVar(0, 1, 0, GRB_BINARY, "b_" + std::to_string(t));
    GRBVar R = master.addVar(0, G, 0, GRB_CONTINUOUS, "R");

    GRBLinExpr obj = n;
    for (int t = 0; t < T_local; t++) obj += b[t];
    obj -= R;
    master.setObjective(obj, GRB_MINIMIZE);

    // Capacity constraints
    for (int t = 0; t < T_local; t++) {
        GRBLinExpr s = 0;
        for (int i = 0; i < n; i++) s += x[t][i];
        master.addConstr(s <= G * b[t], "Capacity_" + std::to_string(t));
    }

    // Test number constraints
    for (int i = 0; i < n; i++) {
        GRBLinExpr s = 0;
        for (int t = 0; t < T_local; t++) s += x[t][i];
        master.addConstr(s <= r, "TestSize_" + std::to_string(i));
    }

    master.set(GRB_IntParam_LazyConstraints, 1);  // Must be set!

    BendersCallback cb(x, b, R, D, &master);
    master.setCallback(&cb);
    master.optimize();

    BDResult res;
    res.x_opt = cb.getXOpt();
    res.b_opt = cb.getBOpt();
    res.ub = cb.getUB();
    res.lb = cb.getLB();
    res.Upperbound = cb.getUpperbound();
    res.status = master.get(GRB_IntAttr_Status);
    res.runtime = master.get(GRB_DoubleAttr_Runtime);
    res.gap = (res.status != GRB_OPTIMAL) ? master.get(GRB_DoubleAttr_MIPGap) : 0.0;
    return res;
}

// ---------------------------------------------------------------------------
// In[5]: iterative (non-callback) Benders decomposition.
// ---------------------------------------------------------------------------
struct BD2Result {
    Matrix x_opt;
    Vector b_opt;
    double Upperbound;
    int status;
    Vector UB;
    Vector LB;
    double Time;
};

BD2Result Bender_decomposition_2(int T_local, const std::vector<Vector>& D) {
    auto start = std::chrono::high_resolution_clock::now();

    int n_local = (int)D[0].size();  // shadows global n, mirrors Python's local reassignment

    GRBEnv env = GRBEnv();
    GRBModel master(env);
    master.set(GRB_IntParam_OutputFlag, 0);
    master.set(GRB_IntParam_LogToConsole, 0);
    master.set(GRB_DoubleParam_TimeLimit, 1800);

    std::vector<std::vector<GRBVar>> x(T_local, std::vector<GRBVar>(n_local));
    std::vector<GRBVar> b(T_local);
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            x[t][i] = master.addVar(0, 1, 0, GRB_BINARY, "x_" + std::to_string(t) + "_" + std::to_string(i));
    for (int t = 0; t < T_local; t++) b[t] = master.addVar(0, 1, 0, GRB_BINARY, "b_" + std::to_string(t));
    GRBVar R = master.addVar(0, G, 0, GRB_CONTINUOUS, "R");

    GRBLinExpr obj = n_local;
    for (int t = 0; t < T_local; t++) obj += b[t];
    obj -= R;
    master.setObjective(obj, GRB_MINIMIZE);

    for (int t = 0; t < T_local; t++) {
        GRBLinExpr s = 0;
        for (int i = 0; i < n_local; i++) s += x[t][i];
        master.addConstr(s <= G * b[t], "Capacity_" + std::to_string(t));
    }
    for (int i = 0; i < n_local; i++) {
        GRBLinExpr s = 0;
        for (int t = 0; t < T_local; t++) s += x[t][i];
        master.addConstr(s <= r, "TestSize_" + std::to_string(i));
    }

    double secondstage_optval = 1e5;
    double Upperbound = 1e5;
    double Lowerbound = n_local + T_local - secondstage_optval;
    Vector LB, UB;
    int iteration = 0;

    Matrix x_fixed, x_fixed_0;
    Vector b_t, b_t_0;

    while (Upperbound - Lowerbound > 0 && iteration < 100) {
        iteration++;

        // Solve master problem and update the lower bound
        master.optimize();
        if (master.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            x_fixed = Matrix(T_local, Vector(n_local));
            for (int t = 0; t < T_local; t++)
                for (int i = 0; i < n_local; i++)
                    x_fixed[t][i] = x[t][i].get(GRB_DoubleAttr_X);

            b_t = Vector(T_local);
            for (int t = 0; t < T_local; t++) b_t[t] = b[t].get(GRB_DoubleAttr_X);

            Lowerbound = master.get(GRB_DoubleAttr_ObjVal);
            LB.push_back(Lowerbound);
        } else {
            break;
        }

        // 1-2. Parallel execution of subproblems, pick the scenario with min objective
        auto [d, opt, p] = solve_all_scenarios_min(x_fixed, T_local, D);
        secondstage_optval = opt;

        // 3. Add optimality cut
        GRBLinExpr cut = buildOptimalityCut(x, d, p);
        master.addConstr(cut >= R, "OptimalityCut_" + std::to_string(iteration));
        master.update();

        // 4. Update the upper bound
        double bsum = 0;
        for (int t = 0; t < T_local; t++) bsum += b_t[t];
        double temp = n_local + bsum - secondstage_optval;
        if (temp < Upperbound) {
            Upperbound = temp;
            x_fixed_0 = x_fixed;
            b_t_0 = b_t;
        }
        UB.push_back(Upperbound);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    double Time = std::chrono::duration<double>(stop - start).count();

    BD2Result res;
    res.x_opt = x_fixed_0;
    res.b_opt = b_t_0;
    res.Upperbound = Upperbound;
    res.status = master.get(GRB_IntAttr_Status);
    res.UB = UB;
    res.LB = LB;
    res.Time = Time;
    return res;
}

// ---------------------------------------------------------------------------
// In[6]: monolithic single-stage MIP (no decomposition).
// ---------------------------------------------------------------------------
struct MIPResult {
    Matrix x_ti;
    Vector b_t;
    double ObjVal;
    double gap;
    int status;
    double Time;
};

MIPResult One_Stage_MIP(int T_local, const std::vector<Vector>& D) {
    double gap = 0;
    int G_local = n;  // Upper bound on total number of items for an individual test

    GRBEnv env = GRBEnv();
    GRBModel model(env);
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_LogToConsole, 0);
    model.set(GRB_DoubleParam_TimeLimit, 1800);

    std::vector<std::vector<GRBVar>> x(T_local, std::vector<GRBVar>(n));
    std::vector<GRBVar> b(T_local);
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            x[t][i] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + std::to_string(t) + "_" + std::to_string(i));
    for (int t = 0; t < T_local; t++) b[t] = model.addVar(0, 1, 0, GRB_BINARY, "b_" + std::to_string(t));
    GRBVar R = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "R");

    std::vector<std::vector<std::vector<GRBVar>>> z_0_ti(T_local, std::vector<std::vector<GRBVar>>(n, std::vector<GRBVar>(K)));
    std::vector<std::vector<std::vector<GRBVar>>> z_1_ti(T_local, std::vector<std::vector<GRBVar>>(n, std::vector<GRBVar>(K)));
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            for (int k = 0; k < K; k++) {
                z_0_ti[t][i][k] = model.addVar(0, 1, 0, GRB_BINARY,
                                                "z_0_ti_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));
                z_1_ti[t][i][k] = model.addVar(0, 1, 0, GRB_BINARY,
                                                "z_1_ti_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));
            }

    std::vector<std::vector<GRBVar>> z_0_i(n, std::vector<GRBVar>(K));
    std::vector<std::vector<GRBVar>> z_1_i(n, std::vector<GRBVar>(K));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++) {
            z_0_i[i][k] = model.addVar(0, 1, 0, GRB_BINARY, "z_0_i_" + std::to_string(i) + "_" + std::to_string(k));
            z_1_i[i][k] = model.addVar(0, 1, 0, GRB_BINARY, "z_1_i_" + std::to_string(i) + "_" + std::to_string(k));
        }

    // Objective function
    GRBLinExpr obj = n;
    for (int t = 0; t < T_local; t++) obj += b[t];
    obj -= R;
    model.setObjective(obj, GRB_MINIMIZE);

    // 1. Capacity constraint
    for (int t = 0; t < T_local; t++) {
        GRBLinExpr s = 0;
        for (int i = 0; i < n; i++) s += x[t][i];
        model.addConstr(s <= G_local * b[t], "Capacity_" + std::to_string(t));
    }

    // 2. Test number constraints
    for (int i = 0; i < n; i++) {
        GRBLinExpr s = 0;
        for (int t = 0; t < T_local; t++) s += x[t][i];
        model.addConstr(s <= r, "TestSize_" + std::to_string(i));
    }

    // 3. R upper bound
    for (int k = 0; k < K; k++) {
        GRBLinExpr s = 0;
        for (int i = 0; i < n; i++) s += z_0_i[i][k] + z_1_i[i][k];
        model.addConstr(R <= s, "R_Upper_Bound_" + std::to_string(k));
    }

    // 4. z_0_ti constraints
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            for (int k = 0; k < K; k++)
                model.addConstr(z_0_ti[t][i][k] <= x[t][i],
                                 "z_0_ti_L1_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            for (int k = 0; k < K; k++)
                for (int j = 0; j < n; j++)
                    model.addConstr(z_0_ti[t][i][k] <= 1 - D[k][j] * x[t][j],
                                     "z_0_ti_L2_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k) + "_" + std::to_string(j));

    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++) {
            GRBLinExpr s = 0;
            for (int t = 0; t < T_local; t++) s += z_0_ti[t][i][k];
            model.addConstr(z_0_i[i][k] <= s, "z_0_i_Sum_" + std::to_string(i) + "_" + std::to_string(k));
        }

    // 5. z_1_ti constraints
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            for (int k = 0; k < K; k++)
                model.addConstr(z_1_ti[t][i][k] <= x[t][i],
                                 "z_1_ti_L1_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            for (int k = 0; k < K; k++) {
                GRBLinExpr s = 0;
                for (int j = 0; j < n; j++) s += D[k][j] * x[t][j];
                model.addConstr(z_1_ti[t][i][k] <= s,
                                 "z_1_ti_L2_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));
            }

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (j != i)
                    for (int k = 0; k < K; k++)
                        model.addConstr(z_1_ti[t][i][k] <= 1 - x[t][j] + z_0_i[j][k],
                                         "z_1_ti_L3_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k));

    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++) {
            GRBLinExpr s = 0;
            for (int t = 0; t < T_local; t++) s += z_1_ti[t][i][k];
            model.addConstr(z_1_i[i][k] <= s, "z_1_i_Sum_" + std::to_string(i) + "_" + std::to_string(k));
        }

    // 6. Solve the model
    model.optimize();

    // 7. Collect the results
    MIPResult res;
    res.x_ti = Matrix(T_local, Vector(n));
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n; i++)
            res.x_ti[t][i] = x[t][i].get(GRB_DoubleAttr_X);

    res.b_t = Vector(T_local);
    for (int t = 0; t < T_local; t++) res.b_t[t] = b[t].get(GRB_DoubleAttr_X);

    res.ObjVal = model.get(GRB_DoubleAttr_ObjVal);
    res.Time = model.get(GRB_DoubleAttr_Runtime);
    res.status = model.get(GRB_IntAttr_Status);
    if (res.status != GRB_OPTIMAL) gap = model.get(GRB_DoubleAttr_MIPGap);
    res.gap = gap;

    return res;
}

// ---------------------------------------------------------------------------
// In[7]: number of items certified purely from group-test logic (no MIP).
// ---------------------------------------------------------------------------
int calc_by_logic(const Matrix& X_in, const Vector& d) {
    int T_local = (int)X_in.size();
    int n_local = (int)X_in[0].size();

    Matrix X(T_local, Vector(n_local));
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            X[t][i] = std::round(X_in[t][i]);
    Matrix X_certify = X;

    Vector test_result(T_local);
    for (int t = 0; t < T_local; t++) {
        double s = 0;
        for (int i = 0; i < n_local; i++) s += X[t][i] * d[i];
        test_result[t] = (s > 0) ? 1.0 : 0.0;
    }

    std::vector<std::string> item_status(n_local, "retest");

    for (int t = 0; t < T_local; t++) {
        if (test_result[t] == 0) {
            for (int i = 0; i < n_local; i++) {
                if (X[t][i] == 1) {
                    for (int tt = 0; tt < T_local; tt++) X_certify[tt][i] = 0;
                    item_status[i] = "non-defective";
                }
            }
        }
    }

    Vector confirming_defective(T_local, 0.0);
    for (int t = 0; t < T_local; t++) {
        double s = 0;
        for (int i = 0; i < n_local; i++) s += X_certify[t][i];
        confirming_defective[t] = s;
    }

    for (int t = 0; t < T_local; t++) {
        if (confirming_defective[t] == 1) {
            for (int i = 0; i < n_local; i++) {
                if (X_certify[t][i] == 1) item_status[i] = "defective";
            }
        }
    }

    int n_certified = 0;
    for (int i = 0; i < n_local; i++)
        if (item_status[i] != "retest") n_certified++;

    return n_certified;
}

// ---------------------------------------------------------------------------
// In[8]: brute-force enumeration over all 2^(T*n) first-stage decisions.
// Intractable at real problem sizes - see the header comment.
// ---------------------------------------------------------------------------
struct EnumResult {
    Matrix x_opt;
    Vector b_opt;
    double first_stage_value;
};

EnumResult enumeration(int T_local, const std::vector<Vector>& D) {
    double first_stage_value = 1e5;
    Matrix x_opt;
    Vector b_opt;

    std::vector<int> bits(T_local * n, 0);
    Matrix x(T_local, Vector(n, 0.0));
    bool done = false;

    while (!done) {
        for (int t = 0; t < T_local; t++)
            for (int i = 0; i < n; i++)
                x[t][i] = bits[t * n + i];

        Vector b(T_local, 0.0);
        for (int t = 0; t < T_local; t++) {
            double s = 0;
            for (int i = 0; i < n; i++) s += x[t][i];
            if (s > 0) b[t] = 1.0;
        }

        // Parallel execution of subproblems (via calc_by_logic, not the MIP subproblem)
        std::vector<std::future<std::pair<Vector, int>>> futures;
        for (const auto& d : D) {
            futures.push_back(std::async(std::launch::async, [x, d]() {
                return std::make_pair(d, calc_by_logic(x, d));
            }));
        }
        std::vector<std::pair<Vector, int>> results;
        for (auto& f : futures) results.push_back(f.get());

        auto minIt = std::min_element(results.begin(), results.end(),
                                       [](const auto& a, const auto& c) { return a.second < c.second; });
        double secondstage_optval = minIt->second;

        double bsum = 0;
        for (int t = 0; t < T_local; t++) bsum += b[t];
        double first_stage_Objval = n + bsum - secondstage_optval;

        if (first_stage_value > first_stage_Objval) {
            first_stage_value = first_stage_Objval;
            x_opt = x;
            b_opt = b;
        }

        // Odometer increment, equivalent to itertools.product([0, 1], repeat=T*n)
        int idx = (int)bits.size() - 1;
        while (idx >= 0 && bits[idx] == 1) {
            bits[idx] = 0;
            idx--;
        }
        if (idx < 0)
            done = true;
        else
            bits[idx] = 1;
    }

    return {x_opt, b_opt, first_stage_value};
}

// ---------------------------------------------------------------------------
// In[9]: driver - generate random scenario sets, run both methods, write CSV.
// ---------------------------------------------------------------------------
int main() {
    try {
        const int iter_max = 1;  // matches Python's `iter_max = 1`
        const int m = n / 2;

        std::vector<double> value_1(iter_max, 1e5), value_2(iter_max, 1e5);
        std::vector<double> Time_1(iter_max, 0.0), Time_2(iter_max, 0.0);
        std::vector<double> gap_1(iter_max, 0.0), gap_2(iter_max, 0.0);
        std::vector<int> status_1(iter_max, 0), status_2(iter_max, 0);

        for (int i = 0; i < iter_max; i++) {
            // NOTE: reproduces the *procedure* of Python's random.seed(i)/random.sample(range(n), m),
            // not the exact bit-for-bit sequence (see header comment).
            std::mt19937 rng(i);
            auto sampleScenario = [&]() {
                std::vector<int> idx(n);
                std::iota(idx.begin(), idx.end(), 0);
                std::shuffle(idx.begin(), idx.end(), rng);
                Vector d(n, 0.0);
                for (int k = 0; k < m; k++) d[idx[k]] = 1.0;
                return d;
            };

            std::vector<Vector> D;
            D.push_back(sampleScenario());
            while ((int)D.size() < K) {
                Vector d = sampleScenario();
                bool duplicate = false;
                for (auto& existing : D) {
                    if (existing == d) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) D.push_back(d);
            }

            BDResult bd = Bender_decomposition(T, D);
            value_1[i] = bd.Upperbound;
            gap_1[i] = bd.gap;
            status_1[i] = bd.status;
            Time_1[i] = bd.runtime;

            MIPResult mip = One_Stage_MIP(T, D);
            value_2[i] = mip.ObjVal;
            gap_2[i] = mip.gap;
            status_2[i] = mip.status;
            Time_2[i] = mip.Time;
        }

        // enumeration(T, D) is intentionally not invoked here, mirroring the
        // commented-out call at the bottom of the Python script.

        std::ostringstream fname;
        fname << "results_K=" << K << "_T=" << T << "_n=" << n << ".csv";
        std::ofstream csv(fname.str());
        csv << ",Optimal Value for Binary Programming,Solution Time (s) for Binary Programming,"
               "Optimal Value for Bender Decomposition,Solution Time (s) for Bender Decomposition\n";
        for (int i = 0; i < iter_max; i++) {
            csv << i << "," << value_2[i] << "," << Time_2[i] << "," << value_1[i] << "," << Time_1[i] << "\n";
        }
        csv.close();

    } catch (GRBException& e) {
        std::cerr << "Gurobi error " << e.getErrorCode() << ": " << e.getMessage() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
