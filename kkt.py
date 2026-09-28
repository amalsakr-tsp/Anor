import math
import random
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
from math import erf, sqrt, exp, pi
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

#def cte(t, H):
    #return (t**H) / (np.sqrt(2 * np.pi))



def cte(t, H, c=70):
    """
    Compute the unconditional expectation E[lambda_{i,t}] for fractional Brownian motion.
    """
    return (t**H) / (np.sqrt(2 * np.pi))


def load(t, type):
    if type == 1:
        a = (62.82023947066044, -31.588509630724403, -13.269685225894834, 5.000838025686271, 1.4776172505340597)
        t_k = (0, 0.4670635411223922, 1.9308223399187476, -9.735350624183834, 0.599138049297667)
        summ = 0
        for k in range(1, len(a)):
            summ += a[k] * np.sin(2 * k * np.pi * (t - t_k[k]) / 24)
        return a[0] + summ
    elif type == 2:
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
    elif type == 4:
        a = (120.82023947066044, -15.588509630724403, 1.269685225894834, 3.000838025686271, 1.4776172505340597)
        t_k = (0, 0.4670635411223922, 1.9308223399187476, -9.735350624183834, 0.599138049297667)
        summ = 0
        for k in range(1, len(a)):
            summ += a[k] * np.sin(2 * k * np.pi * ((t + 10) - t_k[k]) / 24)
        load = a[0] + summ
        return load
    elif type == 5:
        a = (27.12786, 5.408345, 3, 1.9, 3.201077)
        t_k = (0, 47500, 49200, 44700, 45000)
        summ = 0
        for k in range(1, len(a)):
            summ += a[k] * np.sin(2 * k * np.pi * ((t - 15) - t_k[k]) / 24)
        load = a[0] + summ
        return load
    else:
        return 0

def expected_load(t, i, H=0.7):
    return load(t, i) * cte(t, H) * 3600

def optimize_utility_for_timeslot(t, params, C_star):
    # Unpack parameters
    lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots, coalition, players_numb, HC, betas, avg_loads_all_players, horizon, csi = params
    betas = [0] + betas  # Ensure proper indexing for betas
    h = np.zeros(players_numb)  # Initialize h_i values for all SPs
    total_utility = 0
    # Ensure "NO" SP has h = 0
    if (0, "NO") in coalition:
        h[0] = 0  # No allocation for "NO"
        # Calculate numerator for C*
        N = len(coalition) - 1  # Number of active SPs excluding "NO"
        SPs_in_coalition = [l[0] for l in coalition[1:]]  # Skip "NO"
        # Calculate h_i for SPs in the coalition using the given formulas
        for i in SPs_in_coalition:
            if expected_load(t, i) == 0:  # Add constraint: if load is zero, h[i] = 0
                h[i] = 0
            else:
                # Compute geometric mean of beta_j * expected_load(t, j) over all j
                geom_mean = 1
                for j in SPs_in_coalition:
                    geom_mean *= betas[j] * expected_load(t, j)
                geom_mean = geom_mean ** (1 / N)

                # Compute h_star[i]
                numerator = betas[i] * expected_load(t, i)
                h[i] = (C_star / N) + (1 / csi) * np.log(numerator / geom_mean)

        # Calculate total utility for the coalition
        total_utility = sum([
            betas[i] * expected_load(t, i) * (1 - np.exp(-csi * h[i]))
            for i in SPs_in_coalition
        ])

    return total_utility, h



def dichotomy_search_for_optimal_C(lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots,H, coalition, players_numb, HC, betas, avg_loads_all_players, horizon, csi):
    # Get parameters from Foo instance
    params = lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots, coalition, players_numb, HC, betas, avg_loads_all_players, horizon, csi
    foo.set_params(params)
    utilities = np.zeros(players_numb)  # Initialize utilities for all SPs
    total_utility = 0
    betas = [0, 0.0000006, 0.0000006, 0.0000006, 0.0000006, 0.0000006]
    numerator = 0
    optimal_allocations = []  # Store optimal allocations for each time slot
    SPs_in_coalition = [l[0] for l in coalition[1:]]  # Skip "NO"
    h = np.zeros(players_numb)

    if (0, "NO") in coalition:
        h[0] = 0  # Ensure h[0] for "NO" is always 0
        N = len(coalition) - 1
        if N == 0:
            C_star=0
        elif N == 1:  # Check if there's only one SP in the coalition
            i = SPs_in_coalition[0]  # The only SP
            # Compute the sum of squared loads over all time slots
            load_sum = sum(expected_load(t, i) for t in range(daily_timeslots))
            # Ensure valid computation
            if load_sum > 0:
                C_star = - N / csi * np.log(p_cpu / (betas[i] * csi * load_sum))
            else:
                C_star = 0  # Default to 0 if invalid
        else:
            numerator = 0
            for t in range(daily_timeslots):
                product_term = 1
                for i in SPs_in_coalition:
                    product_term *= csi * betas[i] * expected_load(t, i)

                # Geometric mean over all SPs in the coalition
                geometric_mean = product_term ** (1 / N)

                numerator += geometric_mean

            # New denominator

            # Final C_star
            C_star = (N / csi) * np.log(numerator / p_cpu)



    # Loop over daily time slots to compute utility and allocations
        for t in range(daily_timeslots):
            # Solve utility optimization for the current time slot
            if N == 1:
                i = SPs_in_coalition[0]
                if t == 0:  # If it's the first time slot, set h[i] = 0
                    h[i] = 0
                else:  # For all other time slots, set h[i] = C_star
                    h[i] = C_star
                optimal_h = h.copy()  # Store the current allocation
                utility = betas[i] * expected_load(t, i) * (1 - np.exp(-csi * h[i]))
                utilities[i] += utility
            else:
                utility, optimal_h = optimize_utility_for_timeslot(t, params, C_star)
                for i in SPs_in_coalition:
                    utilities[i] += betas[i] * expected_load(t, i) * (1 - np.exp(-csi * optimal_h[i]))

            total_utility += utility
            optimal_allocations.append(optimal_h)

    else:
        C_star = 0  # If "NO" is not in the coalition, default C_star to 0
        optimal_h = [np.zeros(players_numb) for _ in range(daily_timeslots)]
        optimal_allocations.extend(optimal_h)

    #optimal_allocations.append(optimal_h)
    # Save utilities to Foo instance
    foo.set_utilities(utilities)

    # Calculate objective function value
    temp = total_utility - p_cpu * C_star  # Objective function
    return temp, optimal_allocations, C_star, utilities

