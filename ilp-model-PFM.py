"""
Pyomo es un paquete en Python utilizado para formular y resolver problemas de optimización.
El módulo pyomo.environ contiene todas las clases y funciones necesarias para definir
modelos de optimización, variables, restricciones, y funciones objetivo, así como resolver
problemas utilizando diversos solvers. Es el núcleo de Pyomo.
"""
import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
from pyomo.solvers.tests.solvers import initialize
"""
pyomo.dataportal es una herramienta de Pyomo que facilita la carga de datos en modelos
de optimización desde fuentes externas como archivos CSV, Excel, bases de datos, entre otros.
El módulo dataportal ayuda a gestionar datos de entrada de manera más eficiente y accesible.
"""
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización
"""
sys es un módulo estándar de Python que proporciona acceso a algunas variables y funciones
que interactúan con el sistema operativo. Por ejemplo, puedes usarlo para obtener argumentos
de la línea de comandos, manejar excepciones del sistema, o manipular la salida estándar y errores.
"""
import sys # proporciona acceso a funciones relacionadas con el sistema operativo

import pandas as pd # para leer ficheros csv
from pyomo.environ import *


"""
- Hay que tener en cuenta que hay que añadir en algunos momentos la secuencia 0.
- Parámetro i porque aparece "para todo" en la restricción al final!!!!
"""

S_filename = sys.argv[1]
N_filename = sys.argv[2]
C_filename = sys.argv[3]

tau_value = sys.argv[4]

S_content = pd.read_csv(S_filename)

model = pyo.AbstractModel()

model.S = pyo.Set() # Extracted sequences
model.N = pyo.Set(within=model.S*model.S) # Nested sequences
model.C = pyo.Set(within=model.S*model.S) # Conflict sequences

model.x = pyo.Var(model.S, within=pyo.Binary)
model.z = pyo.Var(model.S, model.S, within=pyo.Binary)


model.loc = pyo.Param(model.S, within=pyo.NonNegativeReals) # LOC for each extracted sequence
# model.cc = pyo.Param(model.S, within=pyo.NonNegativeReals) # CC for each extracted sequence - CREO QUE NO HACE FALTA
model.nmcc = pyo.Param(model.S, within=pyo.NonNegativeReals) # New Method Cognitive Complexity
model.ccr = pyo.Param(model.N | model.C, within=pyo.NonNegativeReals) # Cognitive Complexity Reduction

model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=int(tau_value), mutable=True) # Threshold
# model.MAXloc = pyo.Param(within=pyo.NonNegativeReals) # número máximo de líneas de código de todas las secuencias

model.tmax = pyo.Var(within=pyo.NonNegativeReals, initialize=S_content['loc'].max()) # Max LOC
model.tmin = pyo.Var(within=pyo.NonNegativeReals, initialize=S_content['loc'].min()) # min LOC
model.cmax = pyo.Var(within=pyo.NonNegativeReals, initialize=S_content['nmcc'].max()) # Max CC
model.cmin = pyo.Var(within=pyo.NonNegativeReals, initialize=0)



def sequencesObjective(m):
    return sum(m.x[j] for j in m.S)

def LOCdifferenceObjective(m): # modelar segundo objetivo como restricción
    return m.tmax - m.tmin
    
def CCdifferenceObjective(m): # modelar tercer objetivo como restricción
    return m.cmax - m.cmin

def weightedSum(m, sequencesWeight=0.5, LOCdiffWeight=0.5, CCdiffWeight=0.5):
  return sequencesWeight * sequencesObjective(m) + LOCdiffWeight * LOCdifferenceObjective(m) + CCdiffWeight * CCdifferenceObjective(m)

def conflict_sequences(m, i, j): # restricción para las secuencias en conflicto
    return m.x[i] + m.x[j] <= 1

def threshold(m, i): # restricción para no alcanzar el Threshold
    return m.nmcc[i] * m.x[i] - sum((m.ccr[j, k] * m.z[j, k]) for j,k in m.N if k == i) <= m.tau

def zDefinition(m, i, j): # restricción para definir bien las variables z
    interm = [l for l in m.S if (l,j) in m.N and (i,l) in m.N]
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

def x_0(m, i):
    return m.x[0] == 1



model.obj = pyo.Objective(rule=lambda m: weightedSum(m))

# model.min_LOC_difference = pyo.Constraint(model.S, rule=min_LOC_difference) # ESTO SOLO PARA EPSILON-CONSTRAINT
# model.min_CC_difference = pyo.Constraint(model.S, rule=min_CC_difference) # ESTO SOLO PARA EPSILON-CONSTRAINT

model.conflict_sequences = pyo.Constraint(model.C, rule=conflict_sequences)
model.threshold = pyo.Constraint(model.S, rule=threshold)
model.z_definition = pyo.Constraint(model.N, rule=zDefinition)
model.maxLOC = pyo.Constraint(model.S, rule=maxLOC)
model.minLOC = pyo.Constraint(model.S, rule=minLOC)
model.maxCC = pyo.Constraint(model.S, rule=maxCC)
model.minCC = pyo.Constraint(model.S, rule=minCC)
model.x_0 = pyo.Constraint(model.S, rule=x_0)

data = dp.DataPortal()
data.load(filename=S_filename, index=model.S, param=(model.loc, model.nmcc))
data.load(filename=N_filename, index=model.N, param=model.ccr)
data.load(filename=C_filename, index=model.C, param=model.ccr)


concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto

solver = pyo.SolverFactory('cplex')
results = solver.solve(concrete, tee=True)
concrete.pprint()

# Ahora puedes acceder a los valores de las variables
for s in concrete.S:
    print(f"x[{s}] = {concrete.x[s].value}")

if (results.solver.status == 'ok'):
    print('Optimal solution found')
    print('Objective value: ', pyo.value(concrete.obj))
    print('Completion times:')
    for j in concrete.S:
        print(j, pyo.value(concrete.x[j]))
        

