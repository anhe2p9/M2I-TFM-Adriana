import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import algorithms_utils
from typing import Any

import csv
import os

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm


class WeightedSumAlgorithm2obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Weigthed Sum Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal, subs, second_obj) -> list[list[Any]]:
        
        csv_data = [["Weight1","Weight2 (CCdiff)","Num.sequences","CCdif"]]
        
        print(f"ARGS WEIGHTED SUM 2 OBJS: {subs}, {second_obj}")
        
        if isinstance(subs, int):
            print(f"Proccessing all ILP results with {subs} subdivisions")
        
            for i in range(subs+1):
                w1, w2 = algorithms_utils.generate_weights_2obj(subs, i)
                
                _, newrow = algorithms_utils.process_weighted_model_2obj(model, data, w1 ,w2, second_obj)
                
                csv_data.append(newrow)
            
                # if i == 0:
                #     break
        else:
            sys.exit(f'The Weighted Sum Algorithm parameters for two objectives must be a number of subdivisions s and the second objective.')
            
        # Write data in a CSV file.
        filename = "output/output.csv"
        
        if os.path.exists(filename):
            os.remove(filename)
                
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
            print("CSV file correctly created.")
    

    




