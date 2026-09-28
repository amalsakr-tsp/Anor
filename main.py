import os
import time
import matplotlib.pyplot as plt
import numpy as np

import utils
from game import Game

# years until the expiration of the CPU
years = 3
# maximum number of cores that the NO can host
max_cores_hosted = 4
# horizon is the number of days in years

def main(configuration, csi, players_number, price_cpu, horizon, daily_timeslots, H=0.7, chi=0, alpha=0.5,
         HC=max_cores_hosted * 1000, avg_load=1000 * 15,
         heterogeneity_avg_benefit=False, heterogeneity_avg_load=False):
    # starting time to calculate the duration of the simulation
    game = Game()
    print(f"Running simulation with CSI={csi}")

    if not heterogeneity_avg_load:
        avg_loads_all_players = [avg_load] * 2
    else:
        avg_loads_all_players = [(8 / 5) * avg_load, (2 / 5) * avg_load]

    # each coalition element is a tuple player = (id, type)
    rt_players = None
    coalitions = utils.feasible_permutations(players_number, rt_players)

    # list to collect the values of:
    # CPU's capacity
    # px=net benefit, pr=gross benefit, pp=payment for the NO
    y_axis_px_NO = []
    y_axis_pr_NO = []
    y_axis_pp_NO = []
    # px=net benefit, pr=gross benefit, pp=payment for the SP1
    y_axis_px_SP1 = []
    y_axis_pr_SP1 = []
    y_axis_pp_SP1 = []
    # px=net benefit, pr=gross benefit, pp=payment for the SP2
    y_axis_px_SP2 = []
    y_axis_pr_SP2 = []
    y_axis_pp_SP2 = []
    y_coal_payoff = []
    # split of resources among the players
    resources_NO = []
    resources_SP1 = []
    resources_SP2 = []
    resources_SP3 = []
    resources_SP4 = []
    resources_SP5 = []
    resources_SP6 = []
    resources_SP7 = []
    resources_SP8 = []
    resources_SP9 = []
    resources_SP10 = []
    # contribution created from NO, SP1, SP2
    utility_NO = []
    utility_SP1 = []
    utility_SP2 = []
    utility_SP3 = []
    utility_SP4 = []
    utility_SP5 = []
    #utility_SP6= []
    #utility_SP7 = []
    utility_SP8 = []
    utility_SP9 = []
    utility_SP10 = []

    lower_bound = 0.
    upper_bound = 1000
    tolerance = 0.0001
    infos_all_coal_one_config = []
    beta = configuration

    # setting beta for each Service Provider
    if not heterogeneity_avg_benefit:
        betas = [beta] * 2
    else:
        betas = [(8 / 5) * beta, (2 / 5) * beta]

    solution=[]
    all_allocations=[]
    all_coalitions=coalitions
    all_capacities=[]
    # we exclude the empty coalition
    print('c',coalitions)
    for coalition in coalitions[1:]:
        # preparing parameters for the game
        params = (lower_bound, upper_bound, tolerance,
            price_cpu, daily_timeslots, H, coalition, len(coalition), beta, players_number, chi, alpha, HC, betas,
            0, avg_loads_all_players, horizon, csi)

        game.set_params(params)
        # total payoff is the result of the maximization of the objective function v(S)
        sol, optimal_allocations, capacity, utilities = game.calculate_coal_payoff()
        solution.append(sol)
        all_allocations.append(optimal_allocations)
        all_capacities.append(capacity)
        #if coalition == coalitions[-1]:
        grand_coal_payment_one_config = capacity * price_cpu
        total_gross_revenues = sum(utilities)
        print(coalition, utilities)
        # we want to know the contribution just for the grand coalition
        utility_NO.append(horizon * utilities[0])
        utility_SP1.append(horizon * utilities[1])
        utility_SP2.append(horizon * utilities[2])
        #utility_SP3.append(horizon * utilities[3])
        #utility_SP4.append(horizon * utilities[4])
        #utility_SP5.append(horizon * utilities[5])
        #utility_SP6.append(horizon * utilities[6])
        #utility_SP7.append(horizon * utilities[7])
        #utility_SP8.append(horizon * utilities[8])
        #utility_SP9.append(horizon * utilities[9])
        np.random.seed()
        # we store payoffs and the values that optimize the total coalition's payoff
        coal_payoff = sol

        info_one_coalition_one_config = {
            "beta": beta,
            "coalition": coalition,
            "coalitional_payoff": coal_payoff,
        }
        # keeping the payoffs for each coalition to calculate the payoff
        # in fact, we need all the coalitional payoff
        infos_all_coal_one_config.append(info_one_coalition_one_config)
        # keeping the grand coalitional payoff to plot how it changes
        #if coalition == grand_coalition:
        grand_coal_payoff = coal_payoff

        print("Coalition net incomes:", grand_coal_payoff)
        print("Capacity:", capacity, "\n")
        #print("Resources split", optimal_allocations[:24], "\n")

    # result of the game (payoffs' vector) calculated using the Nucleolus
    payoff_vector = game_value_payoffs(infos_all_coal_one_config, players_number, coalitions)#all
    # storing of payoff values for each player to plot them

    print("Nucleolus value is (fair payoff vector):", payoff_vector, "\n")
    print("Proceeding with calculation of revenues vector and payments\n")

    res = game.how_much_rev_paym(payoff_vector, capacity)
    print(res)
    # storing revenues for each player to plot

    print("Revenues array:", res[0], "\n")
    print("Payments array:", res[1], "\n")

    # checking if the calculation of payments and revenues is correct
    print("Checking the correctness of the revenues and payments vectors...\n")
    print("grand_coal_payment_one_config=",grand_coal_payment_one_config)
    print("sum(res[1])=", sum(res[1]))
    print("total_gross_revenues=", total_gross_revenues)
    print("sum(res[0])=", sum(res[0]))

    if abs(grand_coal_payment_one_config - sum(res[1])) > 0.01 or abs(total_gross_revenues - sum(res[0])) > 0.01:
        print("ERROR: the sum of single payments (for each players) or revenues don't match the total "
              "payment/revenues!")
    else:
        print("Total payment and sum of single payments and revenues are correct!\n")


    return solution, all_coalitions, all_allocations, all_capacities, payoff_vector, params

if __name__ == '__main__':
    main()

