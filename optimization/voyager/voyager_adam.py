from optimization.voyager.voyager_problem import VoyagerProblem
from optimization.algorithms.adam import AdamGD

# Whole workflow of opimization with adam

vp = VoyagerProblem()

optimizer = AdamGD(vp, max_iterations=200)

optimizer.optimize()
