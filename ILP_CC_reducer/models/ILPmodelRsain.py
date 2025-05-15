import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización


class ILPmodelRsain(pyo.AbstractModel):
    
    def __init__(self, *args, **kwargs) -> None:
        """Initializes the abstract model."""
        super().__init__(*args,**kwargs)

        self.S = pyo.Set()  # Extracted sequences
        self.N = pyo.Set(within=self.S * self.S)  # Nested sequences
        self.C = pyo.Set(within=self.S * self.S)  # Conflict sequences

        self.loc = pyo.Param(self.S, within=pyo.NonNegativeReals)  # LOC for each extracted sequence
        self.params = pyo.Param(self.S, within=pyo.NonNegativeReals)  # PArameters for each new method
        self.nmcc = pyo.Param(self.S, within=pyo.NonNegativeReals)  # New Method Cognitive Complexity
        self.ccr = pyo.Param(self.N, within=pyo.NonNegativeReals)  # Cognitive Complexity Reduction

        self.x = pyo.Var(self.S, within=pyo.Binary)
        self.z = pyo.Var(self.S, self.S, within=pyo.Binary)

        self.obj = pyo.Objective(rule=lambda m: sequencesObjective(m))

        self.conflict_sequences = pyo.Constraint(self.C, rule=conflict_sequences)
        self.threshold = pyo.Constraint(self.S, rule=threshold)
        self.z_definition = pyo.Constraint(self.N, rule=zDefinition)
        self.x_0 = pyo.Constraint(rule=x_0)

    
    def process_data(self, s_filename: str, n_filename: str, c_filename: str, offsets_filename: str) -> dict:
        
        data = dp.DataPortal()
        
        empty_file = []
        missing_file = []
        if s_filename != "None":
            with open(str(s_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1: # at least there must be one nested sequence
                    data.load(filename=str(s_filename), index=self.S, param=(self.loc, self.nmcc, self.params))
                else:
                    empty_file.append("sequences")
        else:
            missing_file.append("sequences")
            
        
        if n_filename != "None":
            with open(str(n_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1:
                    data.load(filename=str(n_filename), index=self.N, param=self.ccr)
                else:
                    empty_file.append("nested")
        else:
            missing_file.append("nested")
        
        
        if c_filename != "None":
            with open(str(c_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1:
                    data.load(filename=str(c_filename), index=self.C, param=())
                else:
                    empty_file.append("conflict")
        else:
            missing_file.append("conflict")
        
        total_data = {"missingFiles": missing_file, "emptyFiles": empty_file, "data": data, "offsets": offsets_filename}
        # print(f"DATA: {total_data}")
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
        
