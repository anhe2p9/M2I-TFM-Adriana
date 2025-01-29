import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import sys # proporciona acceso a funciones relacionadas con el sistema operativo

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer


class ILPmodelRsain(ILPCCReducer):
    
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
        
        self.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=int(self.tau_value), mutable=True) # Threshold
    
    def define_variables(self):
        """Defines model variables."""
        self.model.x = pyo.Var(self.model.S, within=pyo.Binary)
        self.model.z = pyo.Var(self.model.S, self.model.S, within=pyo.Binary)
    
    def define_constraints(self):
        """Defines model constraints."""
        
        def conflict_sequences(m, i, j): # restricción para las secuencias en conflicto
            return m.x[i] + m.x[j] <= 1
        
        self.model.conflict_sequences = pyo.Constraint(self.model.C, rule=conflict_sequences)
    
        def threshold(m, i): # restricción para no alcanzar el Threshold
            return m.nmcc[i] * m.x[i] - sum((m.ccr[j, i] * m.z[j, i]) for j,k in m.N if k == i) <= m.tau
        
        self.model.threshold = pyo.Constraint(self.model.S, rule=threshold)
        
        def zDefinition(m, j, i): # restricción para definir bien las variables z
            interm = [l for l in m.S if (j,l) in m.N and (l,i) in m.N]
            card_l = len(interm)
            return m.z[j, i] + card_l * (m.z[j, i] - 1) <= m.x[j] - sum(m.x[l] for l in interm)
        
        self.model.z_definition = pyo.Constraint(self.model.N, rule=zDefinition)
        
        def x_0(m):
            return m.x[0] == 1
        
        self.model.x_0 = pyo.Constraint(rule=x_0)
        
        
    def define_objectives(self):
        """Defines objective functions of the model."""

        def sequencesObjective(m):
            return sum(m.x[j] for j in m.S)


# model.obj = pyo.Objective(rule=lambda m: sequencesObjective(m))
#
# model.conflict_sequences = pyo.Constraint(model.C, rule=conflict_sequences)
# model.threshold = pyo.Constraint(model.S, rule=threshold)
# model.z_definition = pyo.Constraint(model.N, rule=zDefinition)
# model.x_0 = pyo.Constraint(rule=x_0) # constraint solo para el primer elemento de model.S
#
# data = dp.DataPortal()
# data.load(filename=S_filename, index=model.S, param=(model.loc, model.nmcc))
# data.load(filename=N_filename, index=model.N, param=model.ccr)
# data.load(filename=C_filename, index=model.C, param=())
#
#
# concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
#
# solver = pyo.SolverFactory('cplex')
# results = solver.solve(concrete)
# concrete.pprint()
#
# num_constraints = sum(len(constraint) for constraint in concrete.component_objects(Constraint, active=True))
# print(f"There are {num_constraints} constraints")
#
# indice = (6,5)  # Índice d ela restricción que quiero ver
# for c in concrete.component_objects(Constraint, active=True):
#     if c.name == "z_definition":
#         print(f"Restricción: {c.name}")
#         for index in c:
#             if index == indice:
#                 print(f"  Índice: {index}, Restricción: {c[index].expr}")
#
#


# for s in concrete.S:
#     print(f"x[{s}] = {concrete.x[s].value}")

# for s in concrete.S:
#     for t in concrete.S:
#         print(f"z[{s, t}] = {concrete.z[s,t].value}")

# if (results.solver.status == 'ok'):
#     print('Optimal solution found')
#     print('Objective value: ', pyo.value(concrete.obj))
#     print('Sequences selected:')
#     # for j in concrete.S:
#     #     print(j, pyo.value(concrete.x[j]))
#     for s in concrete.S:
#         print(f"x[{s}] = {concrete.x[s].value}")
#


