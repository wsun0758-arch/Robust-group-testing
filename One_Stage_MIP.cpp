#include "gurobi_c++.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// ---------------------------------------------------------------------------
// Parameters (module-level globals in the Python source)
// ---------------------------------------------------------------------------
constexpr int K = 3;    // Number of scenarios
constexpr int T = 2;    // Number of tests
constexpr int n = 5;    // Number of entities
constexpr int U = 100;  // Upper bound on total number of tests
constexpr int iter_max = 4;

// ---------------------------------------------------------------------------
// Subproblem (maximization)
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
// One_Stage_MIP
// ---------------------------------------------------------------------------
struct MIPResult {
    Matrix x_ti;
    Vector b_t;
    double ObjVal;
    int status;
};

MIPResult One_Stage_MIP(int T_local, const std::vector<Vector>& D) {
    int n_local = (int)D[0].size();   // Number of items
    int U_local = n_local;            // Upper bound on total number of items for an individual test
    int K_local = (int)D.size();      // Number of scenarios

    GRBEnv env = GRBEnv();
    GRBModel model(env);
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_LogToConsole, 0);

    std::vector<std::vector<GRBVar>> x(T_local, std::vector<GRBVar>(n_local));
    std::vector<GRBVar> b(T_local);
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            x[t][i] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + std::to_string(t) + "_" + std::to_string(i));
    for (int t = 0; t < T_local; t++) b[t] = model.addVar(0, 1, 0, GRB_BINARY, "b_" + std::to_string(t));
    GRBVar R = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "R");

    std::vector<std::vector<std::vector<GRBVar>>> z_0_ti(
        T_local, std::vector<std::vector<GRBVar>>(n_local, std::vector<GRBVar>(K_local)));
    std::vector<std::vector<std::vector<GRBVar>>> z_1_ti(
        T_local, std::vector<std::vector<GRBVar>>(n_local, std::vector<GRBVar>(K_local)));
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int k = 0; k < K_local; k++) {
                z_0_ti[t][i][k] = model.addVar(0, 1, 0, GRB_BINARY,
                                                "z_0_ti_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));
                z_1_ti[t][i][k] = model.addVar(0, 1, 0, GRB_BINARY,
                                                "z_1_ti_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));
            }

    std::vector<std::vector<GRBVar>> z_0_i(n_local, std::vector<GRBVar>(K_local));
    std::vector<std::vector<GRBVar>> z_1_i(n_local, std::vector<GRBVar>(K_local));
    for (int i = 0; i < n_local; i++)
        for (int k = 0; k < K_local; k++) {
            z_0_i[i][k] = model.addVar(0, 1, 0, GRB_BINARY, "z_0_i_" + std::to_string(i) + "_" + std::to_string(k));
            z_1_i[i][k] = model.addVar(0, 1, 0, GRB_BINARY, "z_1_i_" + std::to_string(i) + "_" + std::to_string(k));
        }

    // Objective function: sum(b) - R  (note: no "+n" constant term in this script)
    GRBLinExpr obj = 0;
    for (int t = 0; t < T_local; t++) obj += b[t];
    obj -= R;
    model.setObjective(obj, GRB_MINIMIZE);

    // 1. Capacity constraint
    for (int t = 0; t < T_local; t++) {
        GRBLinExpr s = 0;
        for (int i = 0; i < n_local; i++) s += x[t][i];
        model.addConstr(s <= U_local * b[t], "Capacity_" + std::to_string(t));
    }

    // 2. R upper bound
    for (int k = 0; k < K_local; k++) {
        GRBLinExpr s = 0;
        for (int i = 0; i < n_local; i++) s += z_0_i[i][k] + z_1_i[i][k];
        model.addConstr(R <= s, "R_Upper_Bound_" + std::to_string(k));
    }

    // 3. z_0_ti constraints
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int k = 0; k < K_local; k++)
                model.addConstr(z_0_ti[t][i][k] <= x[t][i],
                                 "z_0_ti_L1_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int k = 0; k < K_local; k++)
                for (int j = 0; j < n_local; j++)
                    model.addConstr(z_0_ti[t][i][k] <= 1 - D[k][j] * x[t][j],
                                     "z_0_ti_L2_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k) + "_" + std::to_string(j));

    for (int i = 0; i < n_local; i++)
        for (int k = 0; k < K_local; k++) {
            GRBLinExpr s = 0;
            for (int t = 0; t < T_local; t++) s += z_0_ti[t][i][k];
            model.addConstr(z_0_i[i][k] <= s, "z_0_i_Sum_" + std::to_string(i) + "_" + std::to_string(k));
        }

    // 4. z_1_ti constraints
    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int k = 0; k < K_local; k++)
                model.addConstr(z_1_ti[t][i][k] <= x[t][i],
                                 "z_1_ti_L1_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int k = 0; k < K_local; k++) {
                GRBLinExpr s = 0;
                for (int j = 0; j < n_local; j++) s += D[k][j] * x[t][j];
                model.addConstr(z_1_ti[t][i][k] <= s,
                                 "z_1_ti_L2_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(k));
            }

    for (int t = 0; t < T_local; t++)
        for (int i = 0; i < n_local; i++)
            for (int j = 0; j < n_local; j++)
                if (j != i)
                    for (int k = 0; k < K_local; k++)
                        model.addConstr(z_1_ti[t][i][k] <= 1 - x[t][j] + z_0_i[j][k],
                                         "z_1_ti_L3_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k));

    for (int i = 0; i < n_local; i++)
        for (int k = 0; k < K_local; k++) {
            GRBLinExpr s = 0;
            for (int t = 0; t < T_local; t++) s += z_1_ti[t][i][k];
            model.addConstr(z_1_i[i][k] <= s, "z_1_i_Sum_" + std::to_string(i) + "_" + std::to_string(k));
        }

    // Solve the model
    model.optimize();

    MIPResult res;
    res.x_ti = Matrix(T_local, Vector(n_local, 0.0));  // matches Python's `x_ti = {(t,i): 0 ...}` default
    res.b_t = Vector(T_local, 1.0);                    // matches Python's `b_t = {t: 1 ...}` default
    res.ObjVal = 1e5;                                  // see header note on the ObjVal NameError bug
    res.status = model.get(GRB_IntAttr_Status);

    // Display the results
    if (res.status == GRB_OPTIMAL) {
        for (int t = 0; t < T_local; t++)
            for (int i = 0; i < n_local; i++)
                res.x_ti[t][i] = x[t][i].get(GRB_DoubleAttr_X);

        for (int t = 0; t < T_local; t++) res.b_t[t] = b[t].get(GRB_DoubleAttr_X);

        double secondstage_optval = 1e5;
        for (const auto& d : D) {
            auto [Q_value, sub_status] = solve_subproblem(res.x_ti, T_local, d);
            if (sub_status == GRB_OPTIMAL) {
                secondstage_optval = std::min(secondstage_optval, Q_value);
            }
        }

        double bsum = 0;
        for (int t = 0; t < T_local; t++) bsum += res.b_t[t];
        res.ObjVal = bsum - secondstage_optval;
    }

    return res;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
int main() {
    try {
        std::vector<double> value_1(iter_max, 1e5);
        std::vector<Matrix> x_ti_1(iter_max);
        std::vector<Vector> b_i_1(iter_max);

        for (int i = 0; i < iter_max; i++) {
            // NOTE: reproduces the *procedure* of numpy's random.seed(i)/random.randint(0, 2, ...),
            // not the exact bit-for-bit sequence (see header comment).
            std::mt19937 rng(i);
            std::bernoulli_distribution coin(0.5);
            auto sampleVector = [&]() {
                Vector d(n);
                for (int j = 0; j < n; j++) d[j] = coin(rng) ? 1.0 : 0.0;
                return d;
            };

            std::vector<Vector> D;
            D.push_back(sampleVector());  // D = random.randint(0, 2, size=(1, n))

            while ((int)D.size() < K) {
                Vector d = sampleVector();

                bool duplicate = false;
                for (auto& existing : D) {
                    if (existing == d) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) D.push_back(d);

                // NOTE: mirrors the Python source's indentation - see header comment. This MIP
                // solve runs on every while-loop iteration (including duplicate-rejected draws),
                // not just once after D reaches size K.
                MIPResult res = One_Stage_MIP(T, D);
                if (res.status == GRB_OPTIMAL) {
                    value_1[i] = res.ObjVal;
                    x_ti_1[i] = res.x_ti;
                    b_i_1[i] = res.b_t;
                }
            }
        }

        for (int i = 0; i < iter_max; i++) {
            std::cout << "iteration " << i << ": ObjVal = " << value_1[i] << std::endl;
        }

    } catch (GRBException& e) {
        std::cerr << "Gurobi error " << e.getErrorCode() << ": " << e.getMessage() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
