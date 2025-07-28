import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización



class MultiobjectiveILPmodel(pyo.AbstractModel):
    
    def __init__(self, *args, **kwargs) -> None:
        """Initializes the abstract model."""
        super().__init__(*args,**kwargs)

        self.S = pyo.Set()  # Extracted sequences
        self.N = pyo.Set(within=self.S*self.S)  # Nested sequences
        self.C = pyo.Set(within=self.S*self.S)  # Conflict sequences

        self.loc = pyo.Param(self.S, within=pyo.NonNegativeReals)  # LOC for each extracted sequence
        self.params = pyo.Param(self.S, within=pyo.NonNegativeReals)  # Parameters for each new method
        self.nmcc = pyo.Param(self.S, within=pyo.NonNegativeReals)  # New Method Cognitive Complexity
        self.ccr = pyo.Param(self.N, within=pyo.NonNegativeReals)  # Cognitive Complexity Reduction

        self.x = pyo.Var(self.S, within=pyo.Binary)
        self.z = pyo.Var(self.S, self.S, within=pyo.Binary)

        self.tmax = pyo.Var(within=pyo.NonNegativeReals)  # Max LOC
        self.tmin = pyo.Var(within=pyo.NonNegativeReals)  # min LOC
        self.cmax = pyo.Var(within=pyo.NonNegativeReals)  # Max CC
        self.cmin = pyo.Var(within=pyo.NonNegativeReals)  # min CC


        self.conflict_sequences = pyo.Constraint(self.C, rule=conflict_sequences)
        self.threshold = pyo.Constraint(self.S, rule=threshold)
        self.z_definition = pyo.Constraint(self.N, rule=z_definition)
        self.max_loc = pyo.Constraint(self.S, rule=max_loc)
        self.min_loc = pyo.Constraint(self.S, rule=min_loc)
        self.max_cc = pyo.Constraint(self.S, rule=max_cc)
        self.min_cc = pyo.Constraint(self.S, rule=min_cc)
        self.x_0 = pyo.Constraint(rule=x_0)
    
    def process_data(self, s_filename: str, n_filename: str, c_filename: str, offsets_filename: str) -> dict:
        
        data = dp.DataPortal()
        
        empty_file = []
        missing_file = []
        if s_filename:
            with open(str(s_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1: # at least there must be one nested sequence
                    data.load(filename=str(s_filename), index=self.S, param=(self.loc, self.nmcc, self.params))
                else:
                    empty_file.append("sequences")
        else:
            missing_file.append("sequences")
            
        
        if n_filename:
            with open(str(n_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1:
                    data.load(filename=str(n_filename), index=self.N, param=self.ccr)
                else:
                    empty_file.append("nested")
        else:
            missing_file.append("nested")
        
        
        if c_filename:
            with open(str(c_filename), 'r', encoding='utf-8') as f:
                if sum(1 for _ in f) > 1:
                    data.load(filename=str(c_filename), index=self.C, param=())
                else:
                    empty_file.append("conflict")
        else:
            missing_file.append("nested")
        
        total_data = {"missingFiles": missing_file, "emptyFiles": empty_file, "data": data, "offsets": offsets_filename}
        # print(f"DATA: {total_data}")
        return total_data
    
    
    @staticmethod
    def extractions_objective(m):
        return sum(m.x[j] for j in m.S)
    
    @staticmethod
    def loc_difference_objective(m): # modelar segundo objetivo como restricción
        return m.tmax - m.tmin
        
    @staticmethod
    def cc_difference_objective(m): # modelar tercer objetivo como restricción
        return m.cmax - m.cmin

    @staticmethod
    def weighted_sum(m, w1, w2, w3, obj1, obj2, obj3):
        return (w1*obj1(m) + w2*obj2(m) + w3*obj3(m))
    
    @staticmethod
    def weighted_sum_2obj(m, w1: int, w2: int, first_objective: pyo.Objective, second_objective: pyo.Objective):
        """ Weighted sum method for two objectives. """
        return w1 * first_objective(m) + w2 * second_objective(m)
    
    @staticmethod
    def weighted_sum_hybrid_method_2objs(m, obj1, obj2):
        return obj1(m) + obj2(m)

    def weighted_sum_hybrid_method(m, obj1, obj2, obj3):
        return obj1(m) + obj2(m) + obj3(m)
    
    @staticmethod
    def epsilon_objective_2obj(m, obj):
        return obj(m) - m.lambda_value * m.s

    @staticmethod
    def epsilon_objective_3obj(m, obj):
        return obj(m) - m.lambda1_value * m.s1 - m.lambda2_value * m.s2

    @staticmethod
    def second_obj_diff_constraint(m, obj):
        return obj(m) <= m.f2z

    @staticmethod
    def epsilon_constraint_2obj(m, obj):
        return obj(m) + m.s == m.epsilon


    



def conflict_sequences(m, i, j): # restricción para las secuencias en conflicto
    return m.x[i] + m.x[j] <= 1

def threshold(m, i): # restricción para no alcanzar el Threshold
    return m.nmcc[i] * m.x[i] - sum((m.ccr[j, i] * m.z[j, i]) for j,k in m.N if k == i) <= m.tau
        
def z_definition(m, j, i): # restricción para definir bien las variables z
    interm = [l for l in m.S if (j,l) in m.N and (l,i) in m.N]
    card_l = len(interm)
    return m.z[j, i] + card_l * (m.z[j, i] - 1) <= m.x[j] - sum(m.x[l] for l in interm)

def max_loc(m, i):
    return m.tmax >= m.loc[i] * m.x[i] - sum((m.loc[j] - 1) * m.z[j, i] for j,k in m.N if k == i)

def min_loc(m, i):
    return m.tmin <= m.loc[0] * (1 - m.x[i]) + m.loc[i] * m.x[i] - sum((m.loc[j] - 1) * m.z[j, k] for j,k in m.N if k == i)
        
def max_cc(m, i):
    return m.cmax >= m.nmcc[i] * m.x[i] - sum(m.ccr[j, i] * m.z[j, i] for j,k in m.N if k == i)

def min_cc(m, i):
    return m.cmin <= (m.tau + 1) * (1 - m.x[i]) + m.nmcc[i] * m.x[i] - sum(m.ccr[j, i] * m.z[j, i] for j,k in m.N if k == i)

def x_0(m):
    return m.x[0] == 1
        