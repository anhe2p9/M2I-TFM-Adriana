import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización
import sys # proporciona acceso a funciones relacionadas con el sistema operativo

import pandas as pd # para leer ficheros csv
from pyomo.environ import *
import math
import utils
import csv
from ILP_operations import ILPOperations


S_filename = sys.argv[1]
N_filename = sys.argv[2]
C_filename = sys.argv[3]

tau_value = sys.argv[4]

class MultiobjectiveILPmodel(ILPOperations):
    
    def __init__(self, data):
        """Initialiczes the model with data."""
        self.model = pyo.AbstractModel()
        self.data = data
        self.define_sets()
        self.define_parameters()
        self.define_variables()
        self.define_constraints()
        self.define_objectives()

    def define_sets(self):
        """Defines model sets."""
        self.model.S = pyo.Set() # Extracted sequences
        self.model.N = pyo.Set(within=self.model.S*self.model.S) # Nested sequences
        self.model.C = pyo.Set(within=self.model.S*self.model.S) # Conflict sequences

    def define_parameters(self):
        """Defines model parameters."""
        self.model.loc = pyo.Param(self.model.S, within=pyo.NonNegativeReals) # LOC for each extracted sequence
        self.model.nmcc = pyo.Param(self.model.S, within=pyo.NonNegativeReals) # New Method Cognitive Complexity
        self.model.ccr = pyo.Param(self.model.N, within=pyo.NonNegativeReals) # Cognitive Complexity Reduction
        
        self.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=int(tau_value), mutable=True) # Threshold
    
    def define_variables(self):
        """Defines model variables."""
        self.model.x = pyo.Var(self.model.S, within=pyo.Binary)
        self.model.z = pyo.Var(self.model.S, self.model.S, within=pyo.Binary)
        
        self.model.tmax = pyo.Var(within=pyo.NonNegativeReals) # Max LOC
        self.model.tmin = pyo.Var(within=pyo.NonNegativeReals) # min LOC
        self.model.cmax = pyo.Var(within=pyo.NonNegativeReals) # Max CC
        self.model.cmin = pyo.Var(within=pyo.NonNegativeReals)

    
    def define_constraints(self):
        """Defines model constraints."""
        
        def conflict_sequences(m, i, j): # restricción para las secuencias en conflicto
            return m.x[i] + m.x[j] <= 1
    
        def threshold(m, i): # restricción para no alcanzar el Threshold
            return m.nmcc[i] * m.x[i] - sum((m.ccr[j, i] * m.z[j, i]) for j,k in m.N if k == i) <= m.tau
        
        def zDefinition(m, j, i): # restricción para definir bien las variables z
            interm = [l for l in m.S if (j,l) in m.N and (l,i) in m.N]
            card_l = len(interm)
            return m.z[j, i] + card_l * (m.z[j, i] - 1) <= m.x[j] - sum(m.x[l] for l in interm)
        
        def maxLOC(m, i):
            return m.tmax >= m.loc[i] * m.x[i] - sum(m.loc[j] * m.z[j, i] for j,k in m.N if k == i)
            
        def minLOC(m, i):
            return m.tmin <= m.loc[0] * (1 - m.x[i]) + m.loc[i] * m.x[i] - sum(m.loc[j] * m.z[j, k] for j,k in m.N if k == i)
            
        def maxCC(m, i):
            return m.cmax >= m.nmcc[i] * m.x[i] - sum(m.ccr[j, i] * m.z[j, i] for j,k in m.N if k == i)
        
        def minCC(m, i):
            return m.cmin <= m.nmcc[0] * (1 - m.x[i]) + m.nmcc[i] * m.x[i] - sum(m.ccr[j, i] * m.z[j, i] for j,k in m.N if k == i)
        
        def x_0(m):
            return m.x[0] == 1
        
        
    def define_objectives(self):
        """Defines objective functions of the model."""

        def sequencesObjective(m):
            return sum(m.x[j] for j in m.S)
        
        def LOCdifferenceObjective(m): # modelar segundo objetivo como restricción
            return m.tmax - m.tmin
            
        def CCdifferenceObjective(m): # modelar tercer objetivo como restricción
            return m.cmax - m.cmin
        
        def weightedSum(m, sequencesWeight, LOCdiffWeight, CCdiffWeight):
            return (sequencesWeight * sequencesObjective(m) +
                    LOCdiffWeight * LOCdifferenceObjective(m) +
                    CCdiffWeight * CCdifferenceObjective(m))
        

    
    
    # Generar las subdivisiones
    # n_divisions = 6
    # theta_div = 2  # índice de 0 a n_divisions-1
    # phi_div = 0  # índice de 0 a n_divisions-1
    # weights = utils.generate_weights(n_divisions, theta_div, phi_div)
    # weights = utils.generate_weights(6,6,6)
    #
    # # print(weights)
    # # print("Variables: ", weights["w1"], weights["w2"], weights["w3"])
    #
    #
    #
    # model.obj = pyo.Objective(rule=lambda m: weightedSum(m, weights["w1"], weights["w2"], weights["w3"]))
    
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
    
    
    # concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
    #
    # solver = pyo.SolverFactory('cplex')
    # results = solver.solve(concrete)
    # concrete.pprint()
    #
    # num_constraints = sum(len(constraint) for constraint in concrete.component_objects(Constraint, active=True))
    # print(f"There are {num_constraints} constraints")
    # if (results.solver.status == 'ok'):
    #     print('Optimal solution found')
    #     print('Objective value: ', pyo.value(concrete.obj))
    #     print('Sequences selected:')
    #     for s in concrete.S:
    #         print(f"x[{s}] = {concrete.x[s].value}")
    
    
csv_data = [["Weight1","Weight2","Weight3","Num.sequences","LOCdif","CCdif"]]

for i in range(7):
    for j in range(7):
        weights = utils.generate_weights(7, j, i)

        if hasattr(model, 'obj'):
            model.del_component('obj')  # Eliminar el componente existente
            model.add_component('obj', pyo.Objective(rule=lambda m: weightedSum(m, weights["w1"], weights["w2"], weights["w3"])))
        else:
            model.obj = pyo.Objective(rule=lambda m: weightedSum(m, weights["w1"], weights["w2"], weights["w3"]))

        concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
        solver = pyo.SolverFactory('cplex')
        results = solver.solve(concrete)

        
        sequences_sum = sum(concrete.x[i].value for i in concrete.S if i != 0)
        
        xLOC = [concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1]
        zLOC = [concrete.loc[j] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
        
        maxLOCselected = abs(max(xLOC) - max(zLOC))
        minLOCselected = min(concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1)
        LOCdif = abs(maxLOCselected - minLOCselected)
        
        
        
        xCC = [concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1]
        zCC = [concrete.ccr[j,ii] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
        
        
        maxCCselected = abs(max(xCC) - max(zCC))
        minCCselected = min(concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1)
        CCdif = abs(maxCCselected - minCCselected)
        
        
        newrow = [round(weights["w1"],2),round(weights["w2"],2),round(weights["w3"],2),sequences_sum,LOCdif,CCdif]
        
        csv_data.append(newrow)
        


print(csv_data)

# Escribir datos en un archivo CSV
with open("C:/Users/X1502/eclipse-workspace/git/M2I-TFM-Adriana/results.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print("Archivo CSV creado correctamente.")
