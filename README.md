Qiskit Knapsack Optimization using QAOA

Project Overview

This project explores the 0/1 Knapsack Optimization Problem using a Quadratic Unconstrained Binary Optimization (QUBO) formulation and the Quantum Approximate Optimization Algorithm (QAOA).

Problem Statement

Select items with given values and weights to maximize total value without exceeding the knapsack capacity.

- Values: [10, 4, 7, 5, 8]
- Weights: [5, 3, 4, 2, 6]
- Capacity: 15

Approach

1. Formulate the knapsack problem as a QUBO objective with a penalty term.
2. Find the classical optimum using brute-force search.
3. Explore QAOA to identify promising solutions.
4. Compare processor constraints and circuit-routing metrics where supported.

Classical Reference Result

- Selected items: 1, 2, 3, 4
- Total value: 26
- Total weight: 14

Project Files

- "main.py" — Main Python program.

How to Run

Install the required dependencies listed in "requirements.txt", then run:

python main.py

Notes

The classical result is a reference for evaluating the quantum approach. Processor routing and noise comparisons should be reported only when the corresponding experiments have actually been run.
