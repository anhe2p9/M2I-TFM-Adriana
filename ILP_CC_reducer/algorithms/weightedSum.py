import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import algorithms_utils

# import csv
# import os

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel


class WeightedSumAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Weigthed Sum algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(data: dp.DataPortal, tau: int, args) -> None:
        
        multiobj_model = MultiobjectiveILPmodel()
    
        # Define threshold
        multiobj_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold
        
        csv_data = [["Weight1","Weight2","Weight3","Num.sequences","CCdif","LOCdif"]]
        
        print(f"ARGS WEIGHTS: {args}")

        concrete = None

        
        """ Weighted model for all generated weights given a number of subdivisions """
        if isinstance(args, int):
            print(f"Proccessing all ILP results with {args} subdivisions")
    
            for i in range(args+1):
                for j in range(args+1):
                    w1, w2, w3 = algorithms_utils.generate_three_weights(args, i, j)
                    
                    _, newrow, _ = process_weighted_model(multiobj_model, data, w1 ,w2, w3)
                    
                    csv_data.append(newrow)
                
                    if i == 0:
                        break
            
            
            """ Weighted model just for a specific given weights """
        elif all(isinstance(arg, float) for arg in args):
            print(f"Processing the optimal ILP solution with weights: {args}")
                        
            w1, w2, w3 = args
            
            concrete, newrow, results = process_weighted_model(multiobj_model, data, w1 ,w2, w3)
                  
            csv_data.append(newrow)
            
            concrete.pprint()
            
            algorithms_utils.print_result_and_sequences(concrete, results.solver.status)
            print(results)
            
        else:
            sys.exit(f'The Weighted Sum algorithm parameters must be a number of subdivisions s or three weights w1,w2,w3.')
        
        
        return csv_data, concrete, None
    



def process_weighted_model(model: pyo.AbstractModel, data: dp.DataPortal, w1 ,w2, w3):
    
    algorithms_utils.modify_component(model, 'obj', pyo.Objective(rule=lambda m: model.weighted_sum(m, w1, w2, w3)))
    concrete, results = algorithms_utils.concrete_and_solve_model(model, data) # para crear una instancia de modelo y hacerlo concreto
    
    newrow_values = [pyo.value(model.sequences_objective(concrete)), pyo.value(model.cc_difference_objective(concrete)), pyo.value(model.loc_difference_objective(concrete))]
    newrow = [round(w1,2),round(w2,2),round(w3,2)] + newrow_values
    
    # TODO: añadir generación de CSVs con los resultados (hay algún método ya hecho creo que sería solo llamarlo)
    
    return concrete, newrow, results






