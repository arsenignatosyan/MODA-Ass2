import pandas as pd
import numpy as np
from desdeo_problem import Variable, ScalarObjective, ScalarConstraint, ScalarMOProblem, variable_builder, MOProblem
from pygmo import non_dominated_front_2d as nd2
import matplotlib.pyplot as plt

# Reads data from a CSV file
data = np.load("output.npy", allow_pickle=True)

# ... data processing
value1 = (max(data[:, 4]) + min(data[:, 4])) / 2

var_names = ["A", "S", "B", "L", "D"]

# Defines initial values, lower bounds, and upper bounds for variables
initial_values = [15, 15, 15, 15, min(data[:, 4]) + 1]
lower_bounds = [14, 14, 14, 14, min(data[:, 4])]
upper_bounds = [58, 58, 58, 58, max(data[:, 4])]

# Creates variables
variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)
print(variables)

# Defines objective functions: Average the sum of scenic beauty, roughness, safety, and slope
def obj1(x):
    y1 = (x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]) / 4
    return y1


# Minimize the distance
def obj2(x):
    y2 = x[:, 4]
    return y2

# Creates objective functions with names and evaluators
f1 = ScalarObjective(name="f1", evaluator=obj1)
f2 = ScalarObjective(name="f2", evaluator=obj2)

# Creates a list of objective functions
list_objs = [f1, f2]

# Defines constraint functions
const_func = lambda x, y: -17 + (x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]) / 4
const_func2 = lambda x, y: value1 - (x[:, 4])

# Creates constraints with names and evaluator functions
cons1 = ScalarConstraint("c_1", 5, 2, const_func)
cons2 = ScalarConstraint("c_2", 5, 2, const_func2)

# Creates a minimization problem
prob = MOProblem(objectives=list_objs, variables=variables, constraints=[cons1, cons2])

# Evaluates the problem with the data
y = prob.evaluate(data[:, :5])

#Pareto Front data
data_pareto = nd2(y.objectives)
y.objectives[data_pareto]


# Extract the Pareto front from y.objectives using the indices obtained from data_pareto
pareto_front = y.objectives[data_pareto]

# Extract objectives for plotting
objective1_pareto = pareto_front[:, 0]
objective2_pareto = pareto_front[:, 1]

# Extract non-dominated solutions (not on the Pareto front)
non_pareto_front = np.delete(y.objectives, data_pareto, axis=0)

# Extract objectives for plotting
objective1_non_pareto = non_pareto_front[:, 0]
objective2_non_pareto = non_pareto_front[:, 1]

# Plot the Pareto front and non-Pareto solutions on the same graph
plt.figure(figsize=(8, 6))
plt.scatter(objective2_pareto, objective1_pareto, color='blue', label='Pareto Front')
plt.scatter(objective2_non_pareto, objective1_non_pareto, color='red', label='Non-Pareto Solutions')
plt.xlabel('obj1')
plt.ylabel('obj2')
plt.title('Pareto Front and Non-Pareto Solutions')
plt.legend()
plt.grid(True)
plt.show()