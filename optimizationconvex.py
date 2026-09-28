import math
import random
import cvxpy as cp
import numpy as np
import concurrent.futures
import os
from fbm import FBM  # Import FBM from the fbm library
from itertools import islice
from multiprocessing import Pool
from scipy.stats import norm


class Foo:

    def get_params(self):
        # Method to get parameters
        return self.params

    def set_params(self, params):
        # Method to set parameters
        self.params = params

    def __init__(self):
        self.utilities = {}

    def get_utilities(self):
        # Method to get utilities
        return self.utilities

    def set_utilities(self, utilities):
        # Method to set utilities
        self.utilities = utilities

foo = Foo()

def _generate_loads(t, type, alpha, fbm_path_t):
    # Define the sinusoidal load generator
    def D(t, load_type):
        if load_type == 1:
            a = (62.82023947066044, -31.588509630724403, -13.269685225894834, 5.000838025686271, 1.4776172505340597)
            t_k = (0, 0.4670635411223922, 1.9308223399187476, -9.735350624183834, 0.599138049297667)
            summ = 0
            for k in range(1, len(a)):
                summ += a[k] * np.sin(2 * k * np.pi * (t - t_k[k]) / 24)
            return a[0] + summ
        elif load_type == 2:
            a = (55.12786, 29.408345, 10.263257, 1.999462, 3.201077)
            t_k = (0, 47500, 49200, 44700, 45000)
            summ = 0
            for k in range(1, len(a)):
                summ += a[k] * np.sin(2 * k * np.pi * ((t - 5) - t_k[k]) / 24)
            return a[0] + summ
        elif type == 3:
            a = (52.12786, 36.408345, 15.263257, 4.999462, 1.201077)
            t_k = (0, 47500, 49200, 44700, 45000)
            summ = 0
            for k in range(1, len(a)):
                summ += a[k] * np.sin(2 * k * np.pi * ((t - 11) - t_k[k]) / 24)
            load = a[0] + summ
            return load
        elif load_type == 33:
            a = (55.12786, 29.408345, 10.263257, 1.999462, 3.201077)
            t_k = (0, 47500, 49200, 44700, 45000)
            summ = 0
            for k in range(1, len(a)):
                summ += a[k] * np.sin(2 * k * np.pi * ((t - 15) - t_k[k]) / 24)
            return a[0] + summ
        else:
            return 0

    sinusoidal_part = D(t,type)  # Sinusoidal modulation
    stochastic_part = alpha * cte(t) + (1 - alpha) * max(0,fbm_path_t)
    A_instantaneous = sinusoidal_part * stochastic_part
    return A_instantaneous
def cte(t, H=0.7):
    return (t**H) / (np.sqrt(2 * np.pi))

def load(t, load_type):
    if load_type == 1:
        a = (62.82023947066044, -31.588509630724403, -13.269685225894834, 5.000838025686271, 1.4776172505340597)
        t_k = (0, 0.4670635411223922, 1.9308223399187476, -9.735350624183834, 0.599138049297667)
        summ = 0
        for k in range(1, len(a)):
            summ += a[k] * np.sin(2 * k * np.pi * (t - t_k[k]) / 24)
        return a[0] + summ
    elif load_type == 2:
        a = (55.12786, 29.408345, 10.263257, 1.999462, 3.201077)
        t_k = (0, 47500, 49200, 44700, 45000)
        summ = 0
        for k in range(1, len(a)):
            summ += a[k] * np.sin(2 * k * np.pi * ((t - 5) - t_k[k]) / 24)
        return a[0] + summ
    elif load_type == 3:
        a = (52.12786, 36.408345, 15.263257, 4.999462, 1.201077)
        t_k = (0, 47500, 49200, 44700, 45000)
        summ = 0
        for k in range(1, len(a)):
            summ += a[k] * np.sin(2 * k * np.pi * ((t - 11) - t_k[k]) / 24)
        load = a[0] + summ
        return load
    else:
        return 0

def expected_load(t, i):
    return load(t, i) * cte(t)

def optimize_utility_for_timeslot(t,C, params):
    lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots, H, coalition, players_numb, HC, betas, loads, horizon, csi = params
    betas = [0] + betas
    h = cp.Variable(players_numb)
    # Check if 'NO' is in the coalition
    if (0, "NO") in coalition:
        # Calculate utility for each player in the coalition
        SPs_in_coalition = [l[0] for l in coalition[1:]]
        total_utility = 0
        for i in SPs_in_coalition:
            total_utility += betas[i] * expected_load(t, i,) * (1 - cp.exp(-csi * h[i])) * 3600
        temp1 = total_utility
        constraints = [cp.sum(h) <= C, h[0] == 0, h >= 0]

        objective = cp.Maximize(temp1)
        problem = cp.Problem(objective, constraints)

        # Try different solvers if one fails
        solvers = [cp.SCS, cp.ECOS, cp.CVXOPT]
        # solvers = [cp.SCS]
        solution_found = False
        for solver in solvers:
            try:
                # Apply specific solver parameters for SCS
                if solver == cp.SCS:
                    problem.solve(solver=cp.SCS, qcp=True)
                else:
                    problem.solve(solver=solver)

                if problem.status not in ["infeasible", "unbounded", "unbounded_inaccurate"]:
                    solution_found = True
                    break
            except cp.error.SolverError as e:
                print(f"Solver Error with {solver}: {e}")

        if solution_found and problem.status not in ["infeasible", "unbounded", "unbounded_inaccurate"]:
            return problem.value, h.value
        else:
            print(f"Optimization problem was {problem.status}.")
            return None, None
    else:
        temp1 = 0.
        utility=0
        h = np.zeros(players_numb)
        return utility, h


def optimize_utility_for_timeslot_worker(args):
    t, C, params = args
    utility, optimal_h = optimize_utility_for_timeslot(t, C, params)
    return t, utility, optimal_h


def compute_F_C(C):
    params = foo.get_params()
    lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots, H, coalition, players_numb, HC, betas, loads, horizon, csi = params
    utilities = [0] * players_numb
    total = 0
    betas = [0] + betas
    optimal_allocations = [None] * (daily_timeslots)  # List to store optimal allocations for each time slot

    # Prepare arguments for parallel processing
    args = [(t, C, params) for t in range(daily_timeslots)]

    # Parallel processing
    with Pool() as pool:
        results = pool.map(optimize_utility_for_timeslot_worker, args)

    # Process results
    for t, utility, optimal_h in results:
        total += utility
        optimal_allocations[t] = optimal_h

        # Correct the utilities calculation
        SPs_in_coalition = [l[0] for l in coalition[1:]]  # Players in the coalition
        for i in SPs_in_coalition:
            # Match player utility calculation with coalition-level utility
            player_utility = betas[i] * expected_load(t, i) * (1 - np.exp(-csi * optimal_h[i])) * 3600
            utilities[i] += player_utility  # Accumulate utilities for each player

    foo.set_utilities(utilities)
    # Compute the final result
    temp = total - p_cpu * C
    return temp, optimal_allocations


def dichotomy_search_for_optimal_C(lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots, H, coalition, players_numb, HC, betas, avg_loads_all_players, horizon, csi):
    params = lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots, H, coalition, players_numb, HC, betas, avg_loads_all_players, horizon, csi
    foo.set_params(params)
    while upper_bound - lower_bound > tolerance:
        mid1 = lower_bound + (upper_bound - lower_bound) / 3
        mid2 = upper_bound - (upper_bound - lower_bound) / 3
        F_C_mid1,_ = compute_F_C(mid1)
        F_C_mid2,_ = compute_F_C(mid2)
        if F_C_mid1 > F_C_mid2:
            upper_bound = mid2
        else:
            lower_bound = mid1

    optimal_C = (lower_bound + upper_bound) / 2
    #optimal_F_C, optimal_allocation = compute_F_C(optimal_C)

    C_prime = optimal_C
    F_C_prime, h_prime = compute_F_C(C_prime)

    return F_C_prime, h_prime,  C_prime, foo.get_utilities()
