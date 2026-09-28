from main import main
from optimizationconvex import _generate_loads
from game import Game
import random
import numpy as np
from scipy.stats import norm
from fbm import FBM  # Import FBM from the fbm library
from kkt import expected_load


class Stochastic_Game:

    def compute_excesses(self, solution, all_coalitions, payoff_vector):
        excess_save = []
        for index, coalition in enumerate(all_coalitions[1:-1], start=1):
            sum_payoffs = sum(payoff_vector[i[0]] for i in coalition)
            excess = sum_payoffs - solution[index - 1]
            excess_save.append(excess)
        return excess_save

    def calculate_stability(self, excess_save):
        eps = min(excess_save)
        stability_value = max(0, eps)
        return stability_value

    def compute_payoff_coalition(self, solution, all_coalitions, stability_value):
        payoff_all_coalitions = []
        for index, coalition in enumerate(all_coalitions[1:-1], start=1):
            payoff_each_coalition = (solution[index-1] + stability_value) / solution[-1]
            difference_in_payoffs = abs(1 - payoff_each_coalition)
            x = len(coalition) * difference_in_payoffs + (len(all_coalitions[-1]) - len(coalition)) * abs(payoff_each_coalition)
            payoff_all_coalitions.append(x)
        return payoff_all_coalitions

    def compute_stochastic(self, all_coalitions, all_allocations, solution, delta_upper):
        def fbm(daily_timeslots, H):
            f = FBM(n=daily_timeslots, hurst=H, length=daily_timeslots, method='daviesharte')
            fbm_path = f.fbm()  # Generate fractional Brownian motion
            return fbm_path
        params = self.get_params()
        _, _, _, price_cpu, daily_timeslots ,H, _, _, beta, players_numb, _, _, _, betas, _, _, horizon, csi = params
        betas = [0] + betas
        stochastic = solution[:]
        infos_all_coal_one_config = []
        delta_t = 3600
        num_iterations = 100  # Number of Monte Carlo iterations
        final_probabilities = []  # Store the final probability for each coalition
        sigmas = [0.6, 0.7, 0.8, 0.9, 1]
        for sigma in sigmas:  # Iterate through each sigma value
            sigma_results = []  # Store probabilities for this sigma

            for index, coalition in enumerate(all_coalitions[1:], start=1):
                print('coalition', coalition)
                sp_probs = []  # To store probabilities for each SP in the coalition
                opt_coal = all_allocations[index - 1]
                if (0, "NO") in coalition and (1, 'rt') in coalition and (2, 'rt') in coalition and (3, 'nrt') in coalition:
                    SPs = [l[0] for l in coalition if l[0] != 0]  # Extract SPs in the coalition
                    z_samples = {i: [] for i in SPs}  # Collect z[i] samples for Monte Carlo

                    for _ in range(num_iterations):  # Monte Carlo iterations
                        z = {i: 0 for i in SPs}  # Initialize z[i] only for SPs in the coalition
                        fbm_paths = {i: fbm(daily_timeslots, H) for i in range(players_numb)}

                        for t in range(daily_timeslots):
                            opt = opt_coal[t]
                            for i in SPs:
                                lambda_iwt= _generate_loads(t, i, sigma, fbm_paths[i][t])
                                expected= expected_load(t, i)
                                z[i] += betas[i] * (lambda_iwt - expected) * (1 - np.exp(-csi * opt[i])) * delta_t
                                z_min_max[i] += np.power(
                                    delta_t * betas[i] * (lamda_max_it - lamda_min_it) * (1 - np.exp(-csi * opt[i])), 2)  # to calculate the probability
                        for i in SPs:
                            z_samples[i].append(z[i])  # Collect z[i] for Monte Carlo

                    if (0, "NO") in coalition:
                        # print('coalition',coalition)
                        SPs = [l[0] for l in coalition if l[0] != 0]
                        for i in SPs:
                            print('SP:', i)
                            # print('z',[i],'minmax', z_min_max[i])
                            proba[i] = 2 * np.exp(- (2 * delta_upper * delta_upper) / z_min_max[i])
                            print('The probability of z', [i], '>= delta upper is:', proba[i])
                            if proba[i] > 1:
                                proba[i] = 1  # Set times to 0 directly if proba[i] > 1
                            p[i] = 1 - proba[i]
                            # print('The probability of z', [i], '< delta upper is:', p[i])
                        print('The probability of all players in the coalition to have zi < delta', coalition, 'is', p)

                        for element in p[1:]:
                            times *= element
                        print('The times probability when sigma =', sigma, 'is:', times)
                    # Calculate probability for each SP in the coalition
                    for i in SPs:
                        prob_i = sum(1 for z_i in z_samples[i] if abs(z_i) < delta_upper) / num_iterations
                        sp_probs.append(prob_i)
                        print(f'  SP {i} probability: {prob_i:.4f}')  # Print probability of each SP
                # Final probability for this coalition
                final_prob = np.prod(sp_probs)  # Product of probabilities of all SPs
                sigma_results.append(final_prob)
                print(f'  Final probability for Coalition {index} with sigma {sigma}: {final_prob:.4f}')  # Print final probability

                stochastic[index - 1] += 0
                info_one_coalition_one_config = {"beta": beta, "coalition": coalition,
                                                 "coalitional_payoff": stochastic[index - 1]}
                infos_all_coal_one_config.append(info_one_coalition_one_config)
            final_probabilities.append(sigma_results)

        print('Solution_stochastic:', stochastic)
        return stochastic, final_probabilities, infos_all_coal_one_config

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params
