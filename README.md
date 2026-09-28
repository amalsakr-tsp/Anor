# Anor

This repository contains the Python code associated with the paper:

**Co-investment under Revenue Uncertainty Based on Stochastic Coalitional Game Theory**  
Amal Sakr, Andrea Araldo, Tijani Chahed, and Daniel Kofman.

Published in **Annals of Operations Research**, 2026.

**Paper:** [https://doi.org/10.1007/s10479-026-07222-w](https://doi.org/10.1007/s10479-026-07222-w)

## Overview

The code implements the co-investment framework studied in the paper. The model is formulated as a stochastic coalitional game between one Infrastructure Provider (InP) and multiple Service Providers (SPs).

The stakeholders jointly invest in Mobile Edge Computing (MEC) infrastructure, while future service revenues are uncertain due to stochastic variations in user demand.

The implementation determines the optimal infrastructure capacity and its allocation among the participating SPs, evaluates coalition values, and allocates coalition payoffs among players.

The code also evaluates the stability and profitability of the grand coalition under revenue uncertainty, including the lower bounds derived in the paper.

## Repository structure

```text
├── mainstochastic.py
├── main2.py
├── game.py
├── gamestochastic.py
├── optimizationconvex.py
├── secondoptkkt.py
├── utils.py
└── requirements.txt
