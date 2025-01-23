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
    return m.nmcc[i] * m.x[i] - sum((m.ccr[j, k] * m.z[j, k]) for j,k in m.N if k == i) <= m.tau

def zDefinition(m, j, i): # restricción para definir bien las variables z
    print("N: ", m.N.pprint())
    print("La j ahora mismo es: ", j)
    for l in m.S:
        print("(j,l): ", (j,l))
        print("(j,l) in m.N: ", (j,l) in m.N)
        print("(l,i): ", (l,i))
        print("(l,i) in m.N: ", (l,i) in m.N)
        print("(j,l) in m.N and (l,i) in m.N: ", (j,l) in m.N and (l,i) in m.N)
    interm = [l for l in m.S if (j,l) in m.N and (l,i) in m.N]
    card_l = len(interm)
    return m.z[j, i] + card_l * (m.z[j, i] - 1) <= m.x[j] - sum(m.x[l] for l in interm)

def x_0(m, i):
    return m.x[0] == 1



model.obj = pyo.Objective(rule=lambda m: sequencesObjective(m))

model.conflict_sequences = pyo.Constraint(model.C, rule=conflict_sequences)
model.threshold = pyo.Constraint(model.S, rule=threshold)
model.z_definition = pyo.Constraint(model.N, rule=zDefinition)
model.x_0 = pyo.Constraint(model.S, rule=x_0)

data = dp.DataPortal()
data.load(filename=S_filename, index=model.S, param=(model.loc, model.nmcc))
data.load(filename=N_filename, index=model.N, param=model.ccr)
data.load(filename=C_filename, index=model.C, param=())


concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto

solver = pyo.SolverFactory('cplex')
results = solver.solve(concrete, tee=True)
concrete.pprint()


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
        

