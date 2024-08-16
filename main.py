from mealpy import FloatVar,GA
from opfunu.cec_based.cec2014 import F112014,F12014
import numpy as np

f_non_convex = F112014() # não convexa
f_convex = F12014() # convexa 



def objective_function(solution, func):
    return func.evaluate(solution)

problem_dict_non_convex = {
    "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    "obj_func": lambda x: objective_function(x, f_non_convex),
    "minmax": "min",
}
problem_dict_convex = {
    "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    "obj_func": lambda x: objective_function(x, f_convex),
    "minmax": "min",
}
def solve_problem(problem_dict):
    model = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection = "random", crossover = "multi_points",
                            mutation = "swap", elite_best = 1, strategy = 0)
    g_best = model.solve(problem_dict)
    return g_best

# Resolvendo para a função não convexa
arq_non_convex = open("arq_non_convex.txt",'w')
g_best_non_convex = solve_problem(problem_dict_non_convex)
arq_non_convex.write(f"Função Não Convexa - Melhor Solução: {g_best_non_convex.solution}, Fitness: {g_best_non_convex.target.fitness}")

# Resolvendo para a função convexa
arq_convex = open("arq_convex.txt",'w')
g_best_convex = solve_problem(problem_dict_convex)
arq_convex.write(f"Função Convexa - Melhor Solução: {g_best_convex.solution}, Fitness: {g_best_convex.target.fitness}")


