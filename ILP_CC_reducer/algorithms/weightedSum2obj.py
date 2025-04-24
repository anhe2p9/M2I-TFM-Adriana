import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import algorithms_utils

import csv
import os

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel


class WeightedSumAlgorithm2obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Weigthed Sum Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(data: dp.DataPortal, tau: int, subs, obj2) -> None:
        
        csv_data = [["Weight1","Weight2 (CCdiff)","Num.sequences","CCdif"]]
        
        multiobj_model = MultiobjectiveILPmodel()
        multiobj_model.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold
                
        
        print(f"ARGS WEIGHTED SUM 2 OBJS: {subs}, {obj2}")
        
        if isinstance(subs, int):
            print(f"Proccessing all ILP results with {subs} subdivisions")
        
            for i in range(subs+1):
                w1, w2 = algorithms_utils.generate_weights_2obj(subs, i)
               
                algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.weightedSum2obj(m, w1, w2, obj2)))
                
                concrete, results = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                
                newrow = algorithms_utils.calculate_results(concrete, obj2)
                            
                algorithms_utils.print_result_and_sequences(results.solver.status, newrow, obj2, concrete)
                
                
                
                csv_data.append(newrow)
                
        else:
            sys.exit(f'The Weighted Sum Algorithm parameters for two objectives must be a number of subdivisions s and the second objective.')
            
        # Write data in a CSV file.
        filename = "output/wheigthed_sum_output.csv"
        
        if os.path.exists(filename):
            os.remove(filename)
                
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
            print("CSV file correctly created.")

    




