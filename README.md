# PSCC_NG_EL_FlexibilityAffinePolicies

PSCC Paper [1] final implementation: Chebyshev Approximation of DRCC and McCormick Envelopes: "main.jl"

Incomplete explorations:
1. Mitchell et al.: Tightening of McCormick envelope: "archives\M2a_Mitchell_Pang_McCormick_enhancement_bounds.jl"
2. Exact Reformulation of DRCC (following approach by Xie et. al. [3]): "archives\M2b_DRCC_ExactSOCReformulation.jl"

#References:

[1] Ratha, Anubhav et al. "Affine Policies for Flexibility Provision by Natural Gas Networks to Power Systems". Proceedings of 21st Power Systems Computation Conference. 2019.

[2]Mitchell and Pang John E. Mitchell, Jong-Shi Pang, and Bin Yu. Convex quadratic relaxations of nonconvex quadratically constrained quadraticprograms.Optimization Methods and Software, 29(1):120–136, 2014.

[3]W. Xie and S. Ahmed. Distributionally robust chance constrained optimal power flow with renewables: A conic reformulation.IEEE Transactions on Power Systems, 33(2):1860–1867, March 2018.


## Code organization:
1. The julia script "main.jl" contains the proposed distributionally robst chance-constrained electricity and natural gas co-optimization problem. The problem takes the form of a second-order cone program which is solved using Mosek solver.
2. The julia script "run_OOS_simulations.jl" evaluates the out-of-sample performance of the proposed methodology. Here, day-ahead decisions including the affine policies allocated are fixed while the dispatch is evaluated for feasibility in the real-time stage when faced with various wind forecast scenarios realized. The uncertain wind forecast datasets are stored in the folder "UncertaintyDataSets".
3. The script "estimate_ex_ante_violations.jl" estimates the ex-ante violation probabilities for the fixed day-ahead outcomes, see [1] for a detailed discussion on their significance.
4. The folder "archives" hosts not only the deterministic co-optimization methods but also incomplete explorations to improve the relaxation exactness of the convex relaxations adopted in [1] using quadratic McCormick enhancements as well as exact SOC reformulation for double-sided chance constraints.
