import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización


class ILPmodelRsain():
    
    def __init__(self):
        """Initializes the abstract model."""
        self.model = pyo.AbstractModel()
        self.defined_model = self.define_model()
        
        
    def define_model(self) -> pyo.AbstractModel:
        """Defines model sets."""

        if not hasattr(self.model, 'S'):
            self.model.S = pyo.Set() # Extracted sequences
            self.model.N = pyo.Set(within=self.model.S*self.model.S) # Nested sequences
            self.model.C = pyo.Set(within=self.model.S*self.model.S) # Conflict sequences
            
            self.model.loc = pyo.Param(self.model.S, within=pyo.NonNegativeReals) # LOC for each extracted sequence
            self.model.params = pyo.Param(self.model.S, within=pyo.NonNegativeReals) # PArameters for each new method
            self.model.nmcc = pyo.Param(self.model.S, within=pyo.NonNegativeReals) # New Method Cognitive Complexity
            self.model.ccr = pyo.Param(self.model.N, within=pyo.NonNegativeReals) # Cognitive Complexity Reduction

            self.model.x = pyo.Var(self.model.S, within=pyo.Binary)
            self.model.z = pyo.Var(self.model.S, self.model.S, within=pyo.Binary)
            
            self.model.tmax = pyo.Var(within=pyo.NonNegativeReals) # Max LOC
            self.model.tmin = pyo.Var(within=pyo.NonNegativeReals) # min LOC
            self.model.cmax = pyo.Var(within=pyo.NonNegativeReals) # Max CC
            self.model.cmin = pyo.Var(within=pyo.NonNegativeReals) # min CC
            
            
            self.model.obj = pyo.Objective(rule=lambda m: sequencesObjective(m))
            
            
            self.model.conflict_sequences = pyo.Constraint(self.model.C, rule=conflict_sequences)
            self.model.threshold = pyo.Constraint(self.model.S, rule=threshold)
            self.model.z_definition = pyo.Constraint(self.model.N, rule=zDefinition)
            self.model.x_0 = pyo.Constraint(rule=x_0)
        
        return self.model
    
    def process_data(self, S_filename: str, N_filename: str, C_filename: str, Offsets_filename: str) -> dict:
        
        data = dp.DataPortal()
        
        empty_file = []
        missing_file = []
        if S_filename != "None":
            with open(str(S_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1: # at least there must be one nested sequence
                    data.load(filename=str(S_filename), index=self.defined_model.S, param=(self.defined_model.loc, self.defined_model.nmcc, self.defined_model.params))
                else:
                    empty_file.append("sequences")
        else:
            missing_file.append("sequences")
            
        
        if N_filename != "None":
            with open(str(N_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1:
                    data.load(filename=str(N_filename), index=self.defined_model.N, param=self.defined_model.ccr)
                else:
                    empty_file.append("nested")
        else:
            missing_file.append("nested")
        
        
        if C_filename != "None":
            with open(str(C_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1:
                    data.load(filename=str(C_filename), index=self.defined_model.C, param=())
                else:
                    empty_file.append("conflict")
        else:
            missing_file.append("conflict")
        
        total_data = {"missingFiles": missing_file, "emptyFiles": empty_file, "data": data, "offsets": Offsets_filename}
        print(f"DATA: {total_data}")
        return total_data
    
    
def sequencesObjective(m):
    return sum(m.x[j] for j in m.S)
    

def conflict_sequences(m, i, j): # restricción para las secuencias en conflicto
    return m.x[i] + m.x[j] <= 1

def threshold(m, i): # restricción para no alcanzar el Threshold
    return m.nmcc[i] * m.x[i] - sum((m.ccr[j, i] * m.z[j, i]) for j,k in m.N if k == i) <= m.tau

def zDefinition(m, j, i): # restricción para definir bien las variables z
    interm = [l for l in m.S if (j,l) in m.N and (l,i) in m.N]
    card_l = len(interm)
    return m.z[j, i] + card_l * (m.z[j, i] - 1) <= m.x[j] - sum(m.x[l] for l in interm)

def x_0(m):
    return m.x[0] == 1
        
        
        

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


