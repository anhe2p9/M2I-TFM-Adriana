import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización
import sys # proporciona acceso a funciones relacionadas con el sistema operativo

import pandas as pd # para leer ficheros csv
from pyomo.environ import *
import math


"""
- Hay que tener en cuenta que hay que añadir en algunos momentos la secuencia 0.
- Parámetro i porque aparece "para todo" en la restricción al final!!!!
"""

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
# model.cc = pyo.Param(model.S, within=pyo.NonNegativeReals) # CC for each extracted sequence - CREO QUE NO HACE FALTA
model.nmcc = pyo.Param(model.S, within=pyo.NonNegativeReals) # New Method Cognitive Complexity
model.ccr = pyo.Param(model.N, within=pyo.NonNegativeReals) # Cognitive Complexity Reduction

model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=int(tau_value), mutable=True) # Threshold
# model.MAXloc = pyo.Param(within=pyo.NonNegativeReals) # número máximo de líneas de código de todas las secuencias

model.tmax = pyo.Var(within=pyo.NonNegativeReals) # Max LOC
model.tmin = pyo.Var(within=pyo.NonNegativeReals) # min LOC
model.cmax = pyo.Var(within=pyo.NonNegativeReals) # Max CC
model.cmin = pyo.Var(within=pyo.NonNegativeReals)



def sequencesObjective(m):
    return sum(m.x[j] for j in m.S)

def LOCdifferenceObjective(m): # modelar segundo objetivo como restricción
    return m.tmax - m.tmin
    
def CCdifferenceObjective(m): # modelar tercer objetivo como restricción
    return m.cmax - m.cmin

def weightedSum(m, sequencesWeight, LOCdiffWeight, CCdiffWeight):
  return sequencesWeight * sequencesObjective(m) + LOCdiffWeight * LOCdifferenceObjective(m) + CCdiffWeight * CCdifferenceObjective(m)

def conflict_sequences(m, i, j): # restricción para las secuencias en conflicto
    return m.x[i] + m.x[j] <= 1

def threshold(m, i): # restricción para no alcanzar el Threshold
    return m.nmcc[i] * m.x[i] - sum((m.ccr[j, i] * m.z[j, i]) for j,k in m.N if k == i) <= m.tau

def zDefinition(m, j, i): # restricción para definir bien las variables z
    interm = [l for l in m.S if (j,l) in m.N and (l,i) in m.N]
    card_l = len(interm)
    return m.z[j, i] + card_l * (m.z[j, i] - 1) <= m.x[j] - sum(m.x[l] for l in interm)

def maxLOC(m, i):
    return m.tmax >= m.loc[i] * m.x[i] - sum(m.loc[j] * m.z[j, k] for j,k in m.N if k == i)
    
def minLOC(m, i):
    return m.tmin <= m.loc[0] * (1 - m.x[i]) + m.loc[i] * m.x[i] - sum(m.loc[j] * m.z[j, k] for j,k in m.N if k == i)
    
def maxCC(m, i):
    return m.cmax >= m.nmcc[i] * m.x[i] - sum(m.ccr[j, k] * m.z[j, k] for j,k in m.N if k == i)

def minCC(m, i):
    return m.cmin <= m.tau * (1 - m.x[i]) + m.nmcc[i] * m.x[i] - sum(m.ccr[j, k] * m.z[j, k] for j,k in m.N if k == i)

def x_0(m):
    return m.x[0] == 1


w1,w2,w3 = 1,1,1

# tita, phi = 0,math.pi
# w1,w2,w3 = math.sin(tita)*math.cos(phi), math.cos(tita)*math.sin(phi), math.cos(tita)
# print("Variables: ", tita, phi, w1, w2, w3)


model.obj = pyo.Objective(rule=lambda m: weightedSum(m, w1, w2, w3))

# model.min_LOC_difference = pyo.Constraint(model.S, rule=min_LOC_difference) # ESTO SOLO PARA EPSILON-CONSTRAINT
# model.min_CC_difference = pyo.Constraint(model.S, rule=min_CC_difference) # ESTO SOLO PARA EPSILON-CONSTRAINT

model.conflict_sequences = pyo.Constraint(model.C, rule=conflict_sequences)
model.threshold = pyo.Constraint(model.S, rule=threshold)
model.z_definition = pyo.Constraint(model.N, rule=zDefinition)
model.maxLOC = pyo.Constraint(model.S, rule=maxLOC)
model.minLOC = pyo.Constraint(model.S, rule=minLOC)
model.maxCC = pyo.Constraint(model.S, rule=maxCC)
model.minCC = pyo.Constraint(model.S, rule=minCC)
model.x_0 = pyo.Constraint(rule=x_0)

data = dp.DataPortal()
data.load(filename=S_filename, index=model.S, param=(model.loc, model.nmcc))
data.load(filename=N_filename, index=model.N, param=model.ccr)
data.load(filename=C_filename, index=model.C, param=())


concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto

solver = pyo.SolverFactory('cplex')
results = solver.solve(concrete)
concrete.pprint()

num_constraints = sum(len(constraint) for constraint in concrete.component_objects(Constraint, active=True))
print(f"There are {num_constraints} constraints")


if (results.solver.status == 'ok'):
    print('Optimal solution found')
    print('Objective value: ', pyo.value(concrete.obj))
    print('Sequences selected:')
    for s in concrete.S:
        print(f"x[{s}] = {concrete.x[s].value}")
        

