import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from pyomo.repn import generate_standard_repn
import numpy as np

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel


class TPAdataAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'TPA data Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains model matrices to apply TPA Algorithm.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal) -> None:
        
        multiobj_model = MultiobjectiveILPmodel()
        #
        # if hasattr(model, 'obj'):
        #     model.del_component('obj')
        # model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.TPAobjective(m)))
        
        concrete = model.create_instance(data)
        concrete.write("output/TPA/model_file.lp", io_options={'symbolic_solver_labels': True})      
        # concrete.write("C:/Users/X1502/eclipse-workspace/git/M2I-TFM-Adriana/output/TPA/model_file.lp")
        
        concrete.pprint()
        
        # Definimos una lista con las funciones objetivo que vamos a usar:
        objectives = [
            multiobj_model.sequencesObjective,
            multiobj_model.LOCdifferenceObjective,
            multiobj_model.CCdifferenceObjective
        ]
        
        coef_rows = []
        
        num_objectives = len(objectives)
        print(f"There are {num_objectives} objectives")
        
        coef_rows.append([num_objectives])
        
        num_variables = len(concrete.S) + len(concrete.N) + 4 # sum(x[i]) + sum(z[j,,i]) + cmax + cmin + tmax + tmin
        print(f"There are {len(concrete.S) + len(concrete.N) + 4} variables")
        
        num_constraints = sum(len(constraint) for constraint in concrete.component_objects(pyo.Constraint, active=True))
        print(f"There are {num_constraints} constraints")
        
        coef_rows.append([num_variables, num_constraints])
        
        
        for obj_func in objectives:
            if hasattr(model, 'obj'):
                model.del_component('obj')
            model.add_component('obj', pyo.Objective(rule=lambda m: obj_func(m)))
            
            concrete = model.create_instance(data)
            
            # Extract variables list and build a id map from id to index
            var_other_objectives = ['tmax', 'tmin', 'cmax', 'cmin']
            var_list = [var for var in list(concrete.component_data_objects(pyo.Var, descend_into=True)) if var.index() in concrete.S or var.index() in concrete.N or var.name in var_other_objectives]
            
            var_to_index = {id(var): idx for idx, var in enumerate(var_list)}
            
            row = np.zeros(num_variables)
            
            repn = generate_standard_repn(concrete.obj)
            
            # Ubicate each coefficient inf the correspondent position
            for coef, var in zip(repn.linear_coefs, repn.linear_vars):
                idx = var_to_index[id(var)]
                row[idx] = coef
            
            row = np.array(row, dtype=int)

            coef_rows.append(row)
        
        filename = f"output/TPA/objective_file"
        
        with open(filename, 'w') as f:
            for row in coef_rows:
                # Convertimos cada número a string y los separamos por espacios
                f.write(' '.join(str(coef) for coef in row) + '\n')
            print("Input files correctly created.")
                
                
                
                
# ./BOXES C:\Users\X1502\eclipse-workspace\git\M2I-TFM-Adriana\output\TPA\objective_file C:\Users\X1502\eclipse-workspace\git\M2I-TFM-Adriana\output\TPA\model_file.lp 0 0 p-partition holzmann reduced_scaled alternate RE 1 1          
        
# ./BOXES /home/novoa/eclipse-workspace/PFMilpPython/output/TPA/objective_file /home/novoa/eclipse-workspace/PFMilpPython/output/TPA/model_file.lp 0 0 p-partition holzmann reduced_scaled alternate RE 1 1
      
           
        
        
        