
#include "gurobi_c++.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Tensor3 = std::vector<Matrix>;

// ---------------------------------------------------------------------------
// Parameters (module-level globals in the Python source)
// ---------------------------------------------------------------------------
constexpr int K = 4;    // Number of scenarios
constexpr int T = 2;    // Number of tests
constexpr int n = 5;    // Number of entities
constexpr int U = 100;  // Upper bound on total number of tests (unused by any model below,
                         // exactly as in the Python source)
constexpr int iter_max = 2;

// ---------------------------------------------------------------------------
// Subproblem (Prime) - LP relaxation of the primal maximization subproblem.
// ---------------------------------------------------------------------------
struct PrimeResult {
    Vector z_0_i_fixed, z_1_i_fixed;      // n
    Matrix z_0_ti_fixed, z_1_ti_fixed;    // T x n
    double objval;
    int status;
};

PrimeResult solve_subproblem_Prime(const Matrix& x_fixed, const Vector& d) {
    GRBEnv env = GRBEnv();
    GRBModel sub(env);
    // NOTE: the Python source never suppresses output here (no OutputFlag/LogToConsole set).

    std::vector<GRBVar> z_0_i(n), z_1_i(n);
    for (int i = 0; i < n; i++) {
        z_0_i[i] = sub.addVar(0, 1, 0, GRB_CONTINUOUS, "z_0_i_" + std::to_string(i));
        z_1_i[i] = sub.addVar(0, 1, 0, GRB_CONTINUOUS, "z_1_i_" + std::to_string(i));
    }

    std::vector<std::vector<GRBVar>> z_0_ti(T, std::vector<GRBVar>(n));
    std::vector<std::vector<GRBVar>> z_1_ti(T, std::vector<GRBVar>(n));
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            z_0_ti[t][i] = sub.addVar(0, 1, 0, GRB_CONTINUOUS, "z_0_ti_" + std::to_string(t) + "_" + std::to_string(i));
            z_1_ti[t][i] = sub.addVar(0, 1, 0, GRB_CONTINUOUS, "z_1_ti_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    GRBLinExpr obj = 0;
    for (int i = 0; i < n; i++) obj += z_0_i[i] + z_1_i[i];
    sub.setObjective(obj, GRB_MAXIMIZE);

    // C1: z_0_ti <= x_fixed
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            sub.addConstr(z_0_ti[t][i] <= x_fixed[t][i], "C1_" + std::to_string(t) + "_" + std::to_string(i));

    // C2: z_0_ti <= 1 - d_j * x_fixed[t,j]
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                sub.addConstr(z_0_ti[t][i] <= 1 - d[j] * x_fixed[t][j],
                              "C2_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));

    // C3: z_0_i <= sum_t z_0_ti
    for (int i = 0; i < n; i++) {
        GRBLinExpr s = 0;
        for (int t = 0; t < T; t++) s += z_0_ti[t][i];
        sub.addConstr(z_0_i[i] <= s, "C3_" + std::to_string(i));
    }

    // C4: z_1_ti <= x_fixed
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            sub.addConstr(z_1_ti[t][i] <= x_fixed[t][i], "C4_" + std::to_string(t) + "_" + std::to_string(i));

    // C5: z_1_ti <= sum_j d_j * x_fixed[t,j]
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += d[j] * x_fixed[t][j];
            sub.addConstr(z_1_ti[t][i] <= s, "C5_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    // C6: z_1_ti <= 1 - x_fixed[t,j] + z_0_i[j], for i != j
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j)
                    sub.addConstr(z_1_ti[t][i] <= 1 - x_fixed[t][j] + z_0_i[j],
                                  "C6_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));

    // C7: z_1_i <= sum_t z_1_ti
    for (int i = 0; i < n; i++) {
        GRBLinExpr s = 0;
        for (int t = 0; t < T; t++) s += z_1_ti[t][i];
        sub.addConstr(z_1_i[i] <= s, "C7_" + std::to_string(i));
    }

    sub.optimize();

    PrimeResult res;
    res.z_0_i_fixed = Vector(n, 0.0);
    res.z_1_i_fixed = Vector(n, 0.0);
    res.z_0_ti_fixed = Matrix(T, Vector(n, 0.0));
    res.z_1_ti_fixed = Matrix(T, Vector(n, 0.0));
    res.objval = 1e5;  // see header note 2: Python would raise a NameError here instead
    res.status = sub.get(GRB_IntAttr_Status);

    if (res.status == GRB_OPTIMAL) {
        for (int i = 0; i < n; i++) {
            res.z_0_i_fixed[i] = z_0_i[i].get(GRB_DoubleAttr_X);
            res.z_1_i_fixed[i] = z_1_i[i].get(GRB_DoubleAttr_X);
        }
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < n; i++) {
                res.z_0_ti_fixed[t][i] = z_0_ti[t][i].get(GRB_DoubleAttr_X);
                res.z_1_ti_fixed[t][i] = z_1_ti[t][i].get(GRB_DoubleAttr_X);
            }
        }
        res.objval = sub.get(GRB_DoubleAttr_ObjVal);
    }

    return res;
}

// ---------------------------------------------------------------------------
// Subproblem (Dual)
// ---------------------------------------------------------------------------
struct DualResult {
    Matrix u_0_ti_fixed, u_1_ti_fixed, u_2_ti_fixed;  // T x n, empty if not optimal
    Tensor3 u_0_tij_fixed;                            // T x n x n, empty if not optimal
    double objval;
    int status;
};

DualResult solve_subproblem_Dual(const Matrix& x_fixed, const Vector& d) {
    // Debug print retained from the Python source's unconditional `print(x_fixed)`.
    std::cout << "x_fixed = [";
    for (int t = 0; t < T; t++) {
        std::cout << "[";
        for (int i = 0; i < n; i++) std::cout << x_fixed[t][i] << (i + 1 < n ? ", " : "");
        std::cout << "]" << (t + 1 < T ? ", " : "");
    }
    std::cout << "]" << std::endl;

    GRBEnv env = GRBEnv();
    GRBModel sub(env);
    sub.set(GRB_IntParam_OutputFlag, 0);

    std::vector<GRBVar> v_0(n), v_1(n), xi(n), eta(n);
    for (int i = 0; i < n; i++) {
        v_0[i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "v_0_" + std::to_string(i));
        v_1[i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "v_1_" + std::to_string(i));
        xi[i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "xi_" + std::to_string(i));
        eta[i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "eta_" + std::to_string(i));
    }

    std::vector<std::vector<GRBVar>> u_0_ti(T, std::vector<GRBVar>(n));
    std::vector<std::vector<GRBVar>> u_1_ti(T, std::vector<GRBVar>(n));
    std::vector<std::vector<GRBVar>> u_2_ti(T, std::vector<GRBVar>(n));
    std::vector<std::vector<GRBVar>> v_0_ti(T, std::vector<GRBVar>(n));
    std::vector<std::vector<GRBVar>> v_1_ti(T, std::vector<GRBVar>(n));
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            u_0_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "u_0_ti_" + std::to_string(t) + "_" + std::to_string(i));
            u_1_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "u_1_ti_" + std::to_string(t) + "_" + std::to_string(i));
            u_2_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "u_2_ti_" + std::to_string(t) + "_" + std::to_string(i));
            v_0_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "v_0_ti_" + std::to_string(t) + "_" + std::to_string(i));
            v_1_ti[t][i] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "v_1_ti_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    std::vector<std::vector<std::vector<GRBVar>>> u_0_tij(T, std::vector<std::vector<GRBVar>>(n, std::vector<GRBVar>(n)));
    std::vector<std::vector<std::vector<GRBVar>>> u_1_tij(T, std::vector<std::vector<GRBVar>>(n, std::vector<GRBVar>(n)));
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                u_0_tij[t][i][j] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS,
                                               "u_0_tij_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));
                u_1_tij[t][i][j] = sub.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS,
                                               "u_1_tij_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
    }

    // Objective function
    GRBLinExpr obj = 0;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            obj += u_0_ti[t][i] * x_fixed[t][i] + u_1_ti[t][i] * x_fixed[t][i];

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += d[j] * x_fixed[t][j];
            obj += u_2_ti[t][i] * s;
        }
    }

    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                obj += u_0_tij[t][i][j] * (1 - d[j] * x_fixed[t][j]);

    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (j != i)
                    obj += u_1_tij[t][i][j] * (1 - x_fixed[t][j]);

    for (int i = 0; i < n; i++) obj += v_0[i] + v_1[i];

    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            obj += v_0_ti[t][i] + v_1_ti[t][i];

    sub.setObjective(obj, GRB_MINIMIZE);

    // Constraint1
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            GRBLinExpr expr = u_0_ti[t][i] + v_0_ti[t][i] - xi[i];
            for (int j = 0; j < n; j++) expr += u_0_tij[t][i][j];
            sub.addConstr(expr >= 0, "Constraint1_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    // Constraint2 - NOTE the index order is u_1_tij[t][j][i], not [t][i][j] (see header note 3)
    for (int i = 0; i < n; i++) {
        GRBLinExpr expr = v_0[i] + xi[i];
        for (int t = 0; t < T; t++)
            for (int j = 0; j < n; j++)
                if (j != i) expr -= u_1_tij[t][j][i];
        sub.addConstr(expr >= 1, "Constraint2_" + std::to_string(i));
    }

    // Constraint3
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            GRBLinExpr expr = u_1_ti[t][i] + u_2_ti[t][i] + v_1_ti[t][i] - eta[i];
            for (int j = 0; j < n; j++)
                if (j != i) expr += u_1_tij[t][i][j];
            sub.addConstr(expr >= 0, "Constraint3_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    // Constraint4
    for (int i = 0; i < n; i++)
        sub.addConstr(v_1[i] + eta[i] >= 1, "Constraint4_" + std::to_string(i));

    sub.optimize();

    // Mirrors Python's unconditional `sub.Objval` read right after optimize().
    double objval = sub.get(GRB_DoubleAttr_ObjVal);
    int status = sub.get(GRB_IntAttr_Status);

    DualResult res;
    res.objval = objval;
    res.status = status;
    // u_*_fixed default to empty containers, matching Python's `[]` default when not optimal.

    if (status == GRB_OPTIMAL) {
        res.u_0_ti_fixed = Matrix(T, Vector(n, 0.0));
        res.u_1_ti_fixed = Matrix(T, Vector(n, 0.0));
        res.u_2_ti_fixed = Matrix(T, Vector(n, 0.0));
        res.u_0_tij_fixed = Tensor3(T, Matrix(n, Vector(n, 0.0)));
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < n; i++) {
                res.u_0_ti_fixed[t][i] = u_0_ti[t][i].get(GRB_DoubleAttr_X);
                res.u_1_ti_fixed[t][i] = u_1_ti[t][i].get(GRB_DoubleAttr_X);
                res.u_2_ti_fixed[t][i] = u_2_ti[t][i].get(GRB_DoubleAttr_X);
                for (int j = 0; j < n; j++) res.u_0_tij_fixed[t][i][j] = u_0_tij[t][i][j].get(GRB_DoubleAttr_X);
            }
        }
    }

    return res;
}

// ---------------------------------------------------------------------------
// Nam_subproblem_Dual - independent dual formulation (constraints 15b-15e).
// ---------------------------------------------------------------------------
struct NamResult {
    double objval;
    int status;
};

NamResult Nam_subproblem_Dual(const Matrix& x_fixed, const Vector& d) {
    GRBEnv env = GRBEnv();
    GRBModel model(env);
    // NOTE: the Python source never suppresses output here either.

    std::vector<std::vector<GRBVar>> lambda_var(T, std::vector<GRBVar>(n));
    std::vector<std::vector<GRBVar>> eta(T, std::vector<GRBVar>(n));
    std::vector<std::vector<GRBVar>> delta(T, std::vector<GRBVar>(n));
    std::vector<std::vector<std::vector<GRBVar>>> gamma(T, std::vector<std::vector<GRBVar>>(n, std::vector<GRBVar>(n)));
    std::vector<std::vector<std::vector<GRBVar>>> kappa(T, std::vector<std::vector<GRBVar>>(n, std::vector<GRBVar>(n)));
    std::vector<GRBVar> xi(n), zeta(n), mu0(n), mu1(n);

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            lambda_var[t][i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "lambda_" + std::to_string(t) + "_" + std::to_string(i));
            eta[t][i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "eta_" + std::to_string(t) + "_" + std::to_string(i));
            delta[t][i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "delta_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                gamma[t][i][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS,
                                               "gamma_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));
                kappa[t][i][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS,
                                               "kappa_" + std::to_string(t) + "_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
    }
    for (int i = 0; i < n; i++) {
        xi[i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "xi_" + std::to_string(i));
        zeta[i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "zeta_" + std::to_string(i));
        mu0[i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "mu0_" + std::to_string(i));
        mu1[i] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "mu1_" + std::to_string(i));
    }

    // Objective (15a)
    GRBLinExpr obj = 0;
    for (int t = 0; t < T; t++) {
        double Ct = 0;
        for (int i1 = 0; i1 < n; i1++) Ct += d[i1] * x_fixed[t][i1];
        for (int i = 0; i < n; i++) {
            obj += x_fixed[t][i] * lambda_var[t][i];
            obj += x_fixed[t][i] * eta[t][i];
            obj += x_fixed[t][i] * Ct * delta[t][i];
        }
    }
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            for (int i1 = 0; i1 < n; i1++)
                obj += (1 - d[i1] * x_fixed[t][i1]) * gamma[t][i][i1];
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (j != i) obj += (1 - x_fixed[t][j]) * kappa[t][i][j];
    for (int i = 0; i < n; i++) obj += mu0[i] + mu1[i];

    model.setObjective(obj, GRB_MINIMIZE);

    // Constraints (15b)
    for (int i = 0; i < n; i++) {
        GRBLinExpr expr = mu0[i] + xi[i];
        for (int t = 0; t < T; t++)
            for (int j = 0; j < n; j++)
                if (j != i) expr -= kappa[t][i][j];
        model.addConstr(expr >= 1, "constr_15b_" + std::to_string(i));
    }

    // Constraints (15c)
    for (int i = 0; i < n; i++)
        model.addConstr(mu1[i] + zeta[i] >= 1, "constr_15c_" + std::to_string(i));

    // Constraints (15d)
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            GRBLinExpr expr = lambda_var[t][i] - xi[i];
            for (int i1 = 0; i1 < n; i1++) expr += gamma[t][i][i1];
            model.addConstr(expr >= 0, "constr_15d_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    // Constraints (15e)
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) {
            GRBLinExpr expr = eta[t][i] + delta[t][i] - zeta[i];
            for (int j = 0; j < n; j++)
                if (j != i) expr += kappa[t][i][j];
            model.addConstr(expr >= 0, "constr_15e_" + std::to_string(t) + "_" + std::to_string(i));
        }
    }

    model.optimize();

    NamResult res;
    res.status = model.get(GRB_IntAttr_Status);
    res.objval = 1e5;  // see header note 4: Python implicitly returns None here instead
    if (res.status == GRB_OPTIMAL) {
        res.objval = model.get(GRB_DoubleAttr_ObjVal);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
int main() {
    try {
        std::vector<std::vector<double>> Q_value_diff_1(iter_max, std::vector<double>(K, 0.0));
        std::vector<std::vector<double>> Q_value_diff_2(iter_max, std::vector<double>(K, 0.0));
        std::vector<std::vector<double>> Q_value_Prime(iter_max, std::vector<double>(K, 0.0));
        std::vector<std::vector<double>> Q_value_Dual(iter_max, std::vector<double>(K, 0.0));
        std::vector<std::vector<double>> Q_value_Nam(iter_max, std::vector<double>(K, 0.0));

        for (int i = 0; i < iter_max; i++) {
            // NOTE: reproduces the *procedure* of numpy's random.seed(i)/random.randint(0, 2, ...),
            // not the exact bit-for-bit sequence (see header note 5).
            std::mt19937 rng(i);
            std::bernoulli_distribution coin(0.5);
            auto sampleVectorN = [&]() {
                Vector d(n);
                for (int j = 0; j < n; j++) d[j] = coin(rng) ? 1.0 : 0.0;
                return d;
            };

            std::vector<Vector> D;
            D.push_back(sampleVectorN());  // D[i] = random.randint(0, 2, size=(1, n))
            while ((int)D.size() < K) {
                Vector d = sampleVectorN();
                bool duplicate = false;
                for (auto& existing : D) {
                    if (existing == d) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) D.push_back(d);
            }

            Matrix x_fixed(T, Vector(n));
            for (int t = 0; t < T; t++)
                for (int j = 0; j < n; j++)
                    x_fixed[t][j] = coin(rng) ? 1.0 : 0.0;  // x_fixed[i] = random.randint(0, 2, size=(T, n))

            // See header note 1: looping per scenario row of D[i] instead of passing the
            // whole D[i] matrix where the Python expected a single length-n vector.
            for (int k = 0; k < K; k++) {
                const Vector& d = D[k];

                PrimeResult prime = solve_subproblem_Prime(x_fixed, d);
                DualResult dual = solve_subproblem_Dual(x_fixed, d);
                NamResult nam = Nam_subproblem_Dual(x_fixed, d);

                Q_value_Prime[i][k] = prime.objval;
                Q_value_Dual[i][k] = dual.objval;
                Q_value_Nam[i][k] = nam.objval;

                if (nam.status == GRB_OPTIMAL && prime.status == GRB_OPTIMAL) {
                    Q_value_diff_1[i][k] = prime.objval - nam.objval;
                }
                if (dual.status == GRB_OPTIMAL && prime.status == GRB_OPTIMAL) {
                    Q_value_diff_2[i][k] = prime.objval - dual.objval;
                }
            }
        }

        for (int i = 0; i < iter_max; i++) {
            for (int k = 0; k < K; k++) {
                std::cout << "iter " << i << " scenario " << k
                          << ": Prime=" << Q_value_Prime[i][k]
                          << " Dual=" << Q_value_Dual[i][k]
                          << " Nam=" << Q_value_Nam[i][k]
                          << " diff1(Prime-Nam)=" << Q_value_diff_1[i][k]
                          << " diff2(Prime-Dual)=" << Q_value_diff_2[i][k]
                          << std::endl;
            }
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
