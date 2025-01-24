import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización
import sys # proporciona acceso a funciones relacionadas con el sistema operativo

import pandas as pd # para leer ficheros csv
from pyomo.environ import *


S_filename = sys.argv[1]
N_filename = sys.argv[2]
C_filename = sys.argv[3]

tau_value = sys.argv[4]

# S_content = pd.read_csv(S_filename)

model = pyo.AbstractModel()

model.S = pyo.Set() # Extracted sequences
model.N = pyo.Set(within=model.S*model.S) # Nested sequences
model.C = pyo.Set(within=model.S*model.S) # Conflict sequences

model.x = pyo.Var(model.S, within=pyo.Binary)
model.z = pyo.Var(model.S, model.S, within=pyo.Binary)


model.loc = pyo.Param(model.S, within=pyo.NonNegativeReals) # LOC for each extracted sequence
model.nmcc = pyo.Param(model.S, within=pyo.NonNegativeReals) # New Method Cognitive Complexity
model.ccr = pyo.Param(model.N, within=pyo.NonNegativeReals) # Cognitive Complexity Reduction

model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=int(tau_value), mutable=True) # Threshold


def sequencesObjective(m):
    return sum(m.x[j] for j in m.S)

def conflict_sequences(m, i, j): # restricción para las secuencias en conflicto
    return m.x[i] + m.x[j] <= 1

def threshold(m, i): # restricción para no alcanzar el Threshold
    # print("nmcc:", [m.nmcc[i] for i in m.S])
    # print("ccr:", {(j, k): m.ccr[j, k] for j,k in m.N if k==i})
    # print("tau:", m.tau.value)
    # print("z: ", [m.z[i].value for i in m.N])
    return m.nmcc[i] * m.x[i] - sum((m.ccr[j, i] * m.z[j, i]) for j,k in m.N if k == i) <= m.tau

def zDefinition(m, j, i): # restricción para definir bien las variables z
    # print("N: ", m.N.pprint())
    # print("La j ahora mismo es: ", j)
    # for l in m.S:
    #     print("(j,l): ", (j,l))
    #     print("(j,l) in m.N: ", (j,l) in m.N)
    #     print("(l,i): ", (l,i))
    #     print("(l,i) in m.N: ", (l,i) in m.N)
    #     print("(j,l) in m.N and (l,i) in m.N: ", (j,l) in m.N and (l,i) in m.N)
    interm = [l for l in m.S if (j,l) in m.N and (l,i) in m.N]
    card_l = len(interm)
    return m.z[j, i] + card_l * (m.z[j, i] - 1) <= m.x[j] - sum(m.x[l] for l in interm)

def x_0(m):
    return m.x[0] == 1



model.obj = pyo.Objective(rule=lambda m: sequencesObjective(m))

model.conflict_sequences = pyo.Constraint(model.C, rule=conflict_sequences)
model.threshold = pyo.Constraint(model.S, rule=threshold)
model.z_definition = pyo.Constraint(model.N, rule=zDefinition)
model.x_0 = pyo.Constraint(rule=x_0) # constraint solo para el primer elemento de model.S

data = dp.DataPortal()
data.load(filename=S_filename, index=model.S, param=(model.loc, model.nmcc))
data.load(filename=N_filename, index=model.N, param=model.ccr)
data.load(filename=C_filename, index=model.C, param=())


concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto

solver = pyo.SolverFactory('cplex')
results = solver.solve(concrete, tee=True)
concrete.pprint()

num_constraints = sum(len(constraint) for constraint in concrete.component_objects(Constraint, active=True))
print(f"There are {num_constraints} constraints")

indice = (6,5)  # Índice d ela restricción que quiero ver
for c in concrete.component_objects(Constraint, active=True):
    if c.name == "z_definition":
        print(f"Restricción: {c.name}")
        for index in c:
            if index == indice:
                print(f"  Índice: {index}, Restricción: {c[index].expr}")
                
                

# for s in concrete.S:
#     print(f"x[{s}] = {concrete.x[s].value}")

# for s in concrete.S:
#     for t in concrete.S:
#         print(f"z[{s, t}] = {concrete.z[s,t].value}")

if (results.solver.status == 'ok'):
    print('Optimal solution found')
    print('Objective value: ', pyo.value(concrete.obj))
    print('Sequences selected:')
    # for j in concrete.S:
    #     print(j, pyo.value(concrete.x[j]))
    for s in concrete.S:
        print(f"x[{s}] = {concrete.x[s].value}")
        

