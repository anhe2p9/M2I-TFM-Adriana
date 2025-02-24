import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización



class MultiobjectiveILPmodel():
    
    def __init__(self):
        """Initializes the abstract model."""
        self.model = pyo.AbstractModel()
        self.defined_model = self.define_model_without_obj()
        


    def define_model_without_obj(self) -> pyo.AbstractModel:
        """Defines model sets."""

        if not hasattr(self.model, 'S'):
            self.model.S = pyo.Set() # Extracted sequences
            self.model.N = pyo.Set(within=self.model.S*self.model.S) # Nested sequences
            self.model.C = pyo.Set(within=self.model.S*self.model.S) # Conflict sequences
            
            self.model.loc = pyo.Param(self.model.S, within=pyo.NonNegativeReals) # LOC for each extracted sequence
            self.model.params = pyo.Param(self.model.S, within=pyo.NonNegativeReals) # Parameters for each new method
            self.model.nmcc = pyo.Param(self.model.S, within=pyo.NonNegativeReals) # New Method Cognitive Complexity
            self.model.ccr = pyo.Param(self.model.N, within=pyo.NonNegativeReals) # Cognitive Complexity Reduction

            self.model.x = pyo.Var(self.model.S, within=pyo.Binary)
            self.model.z = pyo.Var(self.model.S, self.model.S, within=pyo.Binary)
            
            self.model.tmax = pyo.Var(within=pyo.NonNegativeReals) # Max LOC
            self.model.tmin = pyo.Var(within=pyo.NonNegativeReals) # min LOC
            self.model.cmax = pyo.Var(within=pyo.NonNegativeReals) # Max CC
            self.model.cmin = pyo.Var(within=pyo.NonNegativeReals) # min CC
            
            
            self.model.conflict_sequences = pyo.Constraint(self.model.C, rule=conflict_sequences)
            self.model.threshold = pyo.Constraint(self.model.S, rule=threshold)
            self.model.z_definition = pyo.Constraint(self.model.N, rule=zDefinition)
            self.model.maxLOC = pyo.Constraint(self.model.S, rule=maxLOC)
            self.model.minLOC = pyo.Constraint(self.model.S, rule=minLOC)
            self.model.maxCC = pyo.Constraint(self.model.S, rule=maxCC)
            self.model.minCC = pyo.Constraint(self.model.S, rule=minCC)
            self.model.x_0 = pyo.Constraint(rule=x_0)
        
        return self.model
    
    def process_data(self, S_filename: str, N_filename: str, C_filename: str) -> dp.DataPortal:
        
        data = dp.DataPortal()
        data.load(filename=S_filename, index=self.defined_model.S, param=(self.defined_model.loc, self.defined_model.nmcc, self.defined_model.params))
        data.load(filename=N_filename, index=self.defined_model.N, param=self.defined_model.ccr)
        
        with open(C_filename, 'r', encoding='utf-8') as f:
            if sum(1 for _ in f) > 1:
                data.load(filename=str(C_filename), index=self.defined_model.C, param=())
        
        
        return data
    
    
    def sequencesObjective(self, m):
        return sum(m.x[j] for j in m.S)
    
    def LOCdifferenceObjective(self, m): # modelar segundo objetivo como restricción
        return m.tmax - m.tmin
        
    def CCdifferenceObjective(self, m): # modelar tercer objetivo como restricción
        return m.cmax - m.cmin
    
    
    
    
    def weightedSum(self, m, sequencesWeight, LOCdiffWeight, CCdiffWeight):
        return (sequencesWeight * self.sequencesObjective(m) +
                LOCdiffWeight * self.LOCdifferenceObjective(m) +
                CCdiffWeight * self.CCdifferenceObjective(m))
        
    def weightedSum2obj(self, m, sequencesWeight, obj2_Weight, obj: str):
        if obj =='LOC':
            return (sequencesWeight * self.sequencesObjective(m) +
                    obj2_Weight * self.LOCdifferenceObjective(m))
        else:
            return (sequencesWeight * self.sequencesObjective(m) +
                    obj2_Weight * self.CCdifferenceObjective(m))
            
    
    def epsilonObjective(self, m, lambd, obj):
        if obj == 'SEQ':
            return self.sequencesObjective(m) - lambd * m.l
        elif obj == 'LOC':
            return self.LOCdifferenceObjective(m) - lambd * m.l
        else:
            return self.CCdifferenceObjective(m) - lambd * m.l
    

    def epsilonConstraint(self, m, obj):
        if obj == 'SEQ':
            return self.sequencesObjective(m) + m.l == m.epsilon
        elif obj == 'LOC':
            return self.LOCdifferenceObjective(m) + m.l == m.epsilon
        else:
            return self.CCdifferenceObjective(m) + m.l == m.epsilon
        
    def SecondObjdiffConstraint(self, m, obj):
        if obj == 'LOC':
            return self.LOCdifferenceObjective(m) <= m.f2
        else:
            return self.CCdifferenceObjective(m) <= m.f2
    
    def TPAobjective(self, m):
        return sum(m.x[j] for j in m.S) + sum(m.z[j,i] for (j,i) in m.N) + m.tmax + m.tmin + m.cmax + m.cmin
        
        
    
    



def conflict_sequences(m, i, j): # restricción para las secuencias en conflicto
    return m.x[i] + m.x[j] <= 1

def threshold(m, i): # restricción para no alcanzar el Threshold
            return m.nmcc[i] * m.x[i] - sum((m.ccr[j, i] * m.z[j, i]) for j,k in m.N if k == i) <= m.tau
        
def zDefinition(m, j, i): # restricción para definir bien las variables z
            interm = [l for l in m.S if (j,l) in m.N and (l,i) in m.N]
            card_l = len(interm)
            return m.z[j, i] + card_l * (m.z[j, i] - 1) <= m.x[j] - sum(m.x[l] for l in interm)

def maxLOC(m, i):
            return m.tmax >= m.loc[i] * m.x[i] - sum((m.loc[j] - 1) * m.z[j, i] for j,k in m.N if k == i)

def minLOC(m, i):
            return m.tmin <= m.loc[0] * (1 - m.x[i]) + m.loc[i] * m.x[i] - sum((m.loc[j] - 1) * m.z[j, k] for j,k in m.N if k == i)
        
def maxCC(m, i):
            return m.cmax >= m.nmcc[i] * m.x[i] - sum(m.ccr[j, i] * m.z[j, i] for j,k in m.N if k == i)

def minCC(m, i):
            return m.cmin <= (m.tau + 1) * (1 - m.x[i]) + m.nmcc[i] * m.x[i] - sum(m.ccr[j, i] * m.z[j, i] for j,k in m.N if k == i)

def x_0(m):
            return m.x[0] == 1
        