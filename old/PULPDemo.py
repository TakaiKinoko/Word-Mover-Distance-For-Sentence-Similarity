import numpy as np
import pulp

prob = pulp.LpProblem('Manufactuer', sense=pulp.LpMaximize)

X = pulp.LpVariable('X', lowBound=0)
Y = pulp.LpVariable('Y', lowBound=0)

prob += (X+Y-50)
prob += (50*X+24*Y)<=40*60
prob += (30*X+33*Y)<=35*60
prob += X>=45
prob += Y>=5

prob.solve()

pulp.value(prob.objective)

for v in prob.variables():
    print(v.name, ' = ', v.varValue)