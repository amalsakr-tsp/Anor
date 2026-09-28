import math
from kkt import dichotomy_search_for_optimal_C
import cvxpy as cp
from scipy.optimize import minimize
import math
import numpy as np

class Game:

    def calculate_coal_payoff(self):
        params = self.get_params()
        lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots, H, coalition, _, beta, players_numb, chi, alpha, HC, betas, gammas, loads, horizon, csi = params
        sol, optimal_allocations, capacity, utilities = dichotomy_search_for_optimal_C(lower_bound, upper_bound, tolerance, p_cpu, daily_timeslots, H,  coalition, players_numb, HC, betas, loads, horizon, csi)
        return sol, optimal_allocations, capacity, utilities


    def value_payoffs(self, infos_all_coal_one_config, players_number, coalitions):

        # Build coalition value dictionary: v(S)
        tol = 1e-6
        coalition_values = {}
        for coalition_dict in infos_all_coal_one_config:
            coalition_key = frozenset(player_id for player_id, _ in coalition_dict["coalition"])
            coalition_values[coalition_key] = coalition_dict["coalitional_payoff"]

        coalition_values[frozenset()] = 0.0

        # Grand coalition players
        grand_coalition_ids = [player_id for player_id, _ in coalitions[-1]]
        grand_key = frozenset(grand_coalition_ids)

        # Proper coalitions only (exclude empty and grand coalition)
        proper_coalitions = []
        for coalition in coalitions[1:-1]:
            coalition_ids = frozenset(player_id for player_id, _ in coalition)
            if coalition_ids and coalition_ids.issubset(grand_key):
                proper_coalitions.append(coalition_ids)

        # Remove duplicates while preserving order
        proper_coalitions = list(dict.fromkeys(proper_coalitions))

        # Player index map
        player_to_index = {player_id: idx for idx, player_id in enumerate(grand_coalition_ids)}

        x = cp.Variable(players_number)
        fixed_equalities = []   # list of tuples: (coalition_ids, alpha_value)
        last_solution = None

        # Iterative LP for the nucleolus
        for _ in range(players_number + len(proper_coalitions) + 2):
            t = cp.Variable()
            constraints = [cp.sum(x) == coalition_values[grand_key]]

            # Previously binding constraints become equalities
            for coalition_ids, alpha_value in fixed_equalities:
                coeffs = np.zeros(players_number)
                for player_id in coalition_ids:
                    coeffs[player_to_index[player_id]] = 1.0
                constraints.append(coeffs @ x == coalition_values[coalition_ids] + alpha_value)

            active_constraints = []
            for coalition_ids in proper_coalitions:
                if any(coalition_ids == fixed_coalition for fixed_coalition, _ in fixed_equalities):
                    continue

                coeffs = np.zeros(players_number)
                for player_id in coalition_ids:
                    coeffs[player_to_index[player_id]] = 1.0

                # sum_{i in S} x_i - v(S) >= t
                expr = coeffs @ x - coalition_values[coalition_ids]
                constraints.append(expr >= t)
                active_constraints.append((coalition_ids, expr))

            problem = cp.Problem(cp.Maximize(t), constraints)
            problem.solve(solver=cp.SCS)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Nucleolus LP did not solve successfully. Status: {problem.status}")

            last_solution = np.array(x.value).flatten()
            t_star = float(t.value)

            # Detect binding coalitions
            newly_binding = []
            for coalition_ids, expr in active_constraints:
                slack = float(expr.value - t_star)
                if abs(slack) <= max(tol, 10 * tol * (1 + abs(t_star))):
                    newly_binding.append(coalition_ids)

            if not newly_binding:
                break

            added = False
            for coalition_ids in newly_binding:
                if not any(coalition_ids == fixed_coalition for fixed_coalition, _ in fixed_equalities):
                    fixed_equalities.append((coalition_ids, t_star))
                    added = True

            if not added:
                break

            # Stop if equalities already define a unique allocation
            coeff_matrix = [np.ones(players_number)]
            for coalition_ids, _ in fixed_equalities:
                coeffs = np.zeros(players_number)
                for player_id in coalition_ids:
                    coeffs[player_to_index[player_id]] = 1.0
                coeff_matrix.append(coeffs)

            if np.linalg.matrix_rank(np.vstack(coeff_matrix)) >= players_number:
                break

        if last_solution is None:
            raise ValueError("Failed to compute the nucleolus payoff vector.")

        return last_solution.tolist()

    def how_much_rev_paym(self, payoff_vector, capacity):
        # Variables
        _, _, _, p_cpu, _, H,  coalition, _, beta, players_numb, _, _, _, betas, _, _, horizon, csi = self.get_params()

        x = cp.Variable(2 * players_numb)
        # Constraints
        constraints = [
            cp.sum(x[players_numb:]) == p_cpu * capacity,
            x[0] - x[players_numb] == payoff_vector[0],
            x[1] - x[players_numb + 1] == payoff_vector[1],
            x[2] - x[players_numb + 2] == payoff_vector[2],
            #x[3] - x[players_numb + 3] == payoff_vector[3],
            #x[4] - x[players_numb + 4] == payoff_vector[4],
            #x[5] - x[players_numb + 5] == payoff_vector[5],
            #x[6] - x[players_numb + 6] == payoff_vector[6],
            #x[7] - x[players_numb + 7] == payoff_vector[7],
            #x[8] - x[players_numb + 8] == payoff_vector[8],
            #x[9] - x[players_numb + 9] == payoff_vector[9]
        ]

        # Objective (For a feasibility problem, it's a dummy objective)
        objective = cp.Maximize(0)

        problem = cp.Problem(objective, constraints)
        # Try different solvers if one fails
        solvers = [cp.SCS, cp.ECOS, cp.CVXOPT]
        #solvers = [cp.SCS]
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
            return x.value[0:players_numb], x.value[players_numb:]
        else:
            print(f"Optimization problem two was {problem.status}.")
            return 0
    # GETTERS AND SETTERS
    # to get parameters p_cpu, horizon, coalition, players_number
    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params
