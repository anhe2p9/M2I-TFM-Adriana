import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer


class MultiobjectiveILPmodel(ILPCCReducer):
    
    def __init__(self, tau_value: int):
        """Initializes the abstract model."""
        self.tau_value = tau_value
        
        self.model = pyo.AbstractModel()
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
        
        self.model.tmax = pyo.Var(within=pyo.NonNegativeReals) # Max LOC
        self.model.tmin = pyo.Var(within=pyo.NonNegativeReals) # min LOC
        self.model.cmax = pyo.Var(within=pyo.NonNegativeReals) # Max CC
        self.model.cmin = pyo.Var(within=pyo.NonNegativeReals)

    
    def define_constraints(self):
        """Defines model constraints."""

        self.model.conflict_sequences = pyo.Constraint(self.model.C, rule=conflict_sequences)
        self.model.threshold = pyo.Constraint(self.model.S, rule=threshold)
        self.model.z_definition = pyo.Constraint(self.model.N, rule=zDefinition)
        self.model.maxLOC = pyo.Constraint(self.model.S, rule=maxLOC)
        self.model.minLOC = pyo.Constraint(self.model.S, rule=minLOC)
        self.model.maxCC = pyo.Constraint(self.model.S, rule=maxCC)
        self.model.minCC = pyo.Constraint(self.model.S, rule=minCC)
        self.model.x_0 = pyo.Constraint(rule=x_0)
        
        
    def define_objectives(self):
        """Defines objective functions of the model."""
        pass
    
    def process_data(self, S_filename: str, N_filename: str, C_filename: str) -> dp.DataPortal:
        
        data = dp.DataPortal()
        data.load(filename=S_filename, index=self.model.S, param=(self.model.loc, self.model.nmcc))
        data.load(filename=N_filename, index=self.model.N, param=self.model.ccr)
        data.load(filename=C_filename, index=self.model.C, param=())
        
        return data

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
            return m.cmin <= (m.tau + 1) * (1 - m.x[i]) + m.nmcc[i] * m.x[i] - sum(m.ccr[j, i] * m.z[j, i] for j,k in m.N if k == i)

def x_0(m):
            return m.x[0] == 1
        