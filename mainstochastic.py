from gamestochastic import Stochastic_Game
from main2 import main
from game import Game
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import seaborn as sns
import io
import psutil
import sys

# Directory where we want to save the plots
save_dir = r"C:\Users\asakr\Desktop\test"
# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Path for the output log file
output_file_path = os.path.join(save_dir, 'output.txt')
sys.stdout = open(output_file_path, 'w')

def log_resource_usage():
    # Log CPU and memory usage
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    print(f"CPU usage: {cpu_usage}%")
    print(f"Memory usage: {memory_info.percent}%")
    print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")

def main_logic(csi=0.03, players_number=6, horizon=1):  # 0.0001=csi
    start = time.time()
    beta_centr = 0.0000006
    print(f"Running simulation with benefit factor={beta_centr}")
    print('Players number:', players_number)

    configurations = [beta_centr]
    y_axis_cpu_capacity = []
    y_coal_payoff = []
    probabilities = []
    price_cpu = 10.94 + 16.25 * (12 * 5)  # PER month
    Daily_timeslots = [24 * 30 * 12 * 5]  # 5 * 365 * number of time slots in one day
    for daily_timeslots in Daily_timeslots:
        print(f"Running simulation with daily_timeslot={daily_timeslots}")
        log_resource_usage()  # Log resources at the start
        solution, all_coalitions, all_allocations, all_capacities, payoff_vector, params = main(
            beta_centr, csi, players_number, price_cpu, horizon, daily_timeslots
        )
        (_, _, _, price_cpu, daily_timeslots, H, _, _, beta, players_numb, _, _, _, betas, _, _, horizon, csi) = params

        y_axis_cpu_capacity.append(all_capacities[-1])
        y_coal_payoff.append(solution[-1])

        game = Game()
        StochasticGame = Stochastic_Game()
        StochasticGame.set_params(params)
        game.set_params(params)

        excess_save = StochasticGame.compute_excesses(solution, all_coalitions, payoff_vector)
        print('Excess:', excess_save)
        stability_value = StochasticGame.calculate_stability(excess_save)
        print("Stability Value:", stability_value)

        payoff_all_coalitions = StochasticGame.compute_payoff_coalition(solution, all_coalitions, stability_value)
        print('Payoff all coalitions:', payoff_all_coalitions)
        maximum = max(payoff_all_coalitions)
        print("Maximum payoff coalition:", maximum)

        delta_upper = min(solution[-1] / players_numb, stability_value / maximum)
        print("Delta Upper Bound:", delta_upper)

        # Print all allocations for all coalitions, including the final one
        print("Total number of allocations:", len(all_allocations))
        for idx, allocation in enumerate(all_allocations):
            print(f"Coalition {idx + 1}: {allocation}")

        #print('Solution_deterministic:', solution)
        #log_resource_usage()  # Log resources after each simulation

    #print("Time required for the simulation: ", round(time.time() - start), "seconds")
    #print('Coalition 7:', all_allocations[-1])
    return all_allocations[-1], all_capacities[-1], params


if __name__ == "__main__":
    main_logic()
sys.stdout.close()
