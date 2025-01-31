import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from pyomo.repn import generate_standard_repn
import numpy as np

from typing import Any

from ILP_CC_reducer.CC_reducer.ILP_CCreducer import ILPCCReducer


class TPAdataAlgorithm(ILPCCReducer):

    @staticmethod
    def get_name() -> str:
        return 'TPA data Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains model matrices to apply TPA Algorithm.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal) -> Any:
        
        print(f"Proccessing ILP matrix and row vector for Ax <= b standard form.")
        
        concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
        
        # 1. Obtain variable list and assign an index for each one using id() key
        var_list = list(concrete.component_data_objects(pyo.Var, descend_into=True))
        var_to_index = {id(var): idx for idx, var in enumerate(var_list)}
        
        num_vars = len(var_list)
        # Initialize lists to save matrix rows and independent term
        A_rows = []  # Coefficients matrix rows
        indep_terms = []     # Right-hand side (independent terms)
        senses = []  # Constrints senses (e.g., '=', '<=', '>=')
        
        # 2. Scroll through the active constraints
        for constr in concrete.component_data_objects(pyo.Constraint, active=True):
            # generate_standard_repn: standard representation of constraint expression
            repn = generate_standard_repn(constr.body)
            
            # Verify that constraint is linear
            if repn is None or not repn.linear_vars:
                print(f"Constraint {constr.name} is not linear and will me omitted.")
                continue
            
            # Initialize a zeros row for A matrix
            row = np.zeros(num_vars)
            
            # For each linear variable and their coefficients, ubicate the coefficient in the correspondent position
            for coef, var in zip(repn.linear_coefs, repn.linear_vars):
                idx = var_to_index[id(var)]
                row[idx] = coef
        
            # Process depending of the constraint type
            if constr.equality:
                A_rows.append(row)
                b_val = constr.lower - repn.constant
                indep_terms.append(b_val)
                senses.append('=')
            elif constr.has_ub() and not constr.has_lb():
                A_rows.append(row)
                b_val = constr.upper - repn.constant
                indep_terms.append(b_val)
                senses.append('<=')
            elif constr.has_lb() and not constr.has_ub():
                A_rows.append(row)
                b_val = constr.lower - repn.constant
                indep_terms.append(b_val)
                senses.append('>=')
            elif constr.has_lb() and constr.has_ub():
                # If restricted constraint, it generates one row for each limit
                A_rows.append(row)
                b_val_lower = constr.lower - repn.constant
                indep_terms.append(b_val_lower)
                senses.append('>=')
                
                A_rows.append(row.copy())
                b_val_upper = constr.upper - repn.constant
                indep_terms.append(b_val_upper)
                senses.append('<=')
            else:
                print(f"Constraint type not recognized in {constr.name}.")
        
        # Change the rows list into a NumPy matrix
        # A = np.array(A_rows)
        
        # Show coefficients matrix and independent terms vector.
        # print("Matriz de coeficientes (A):")
        # print(A)
        # print("\nVector del lado derecho (b):")
        # print(indep_terms)
        
        return A_rows, indep_terms