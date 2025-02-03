import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import utils
from typing import Any

import csv
import os

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm


class WeightedSumAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Weigthed Sum Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(model: pyo.AbstractModel, data: dp.DataPortal, *args) -> list[list[Any]]:
        
        csv_data = [["Weight1","Weight2","Weight3","Num.sequences","LOCdif","CCdif"]]
        
        args = args[0]
        
 
        if isinstance(args, int):
            print(f"Proccessing all ILP results with {args} subdivisions")
    
            for i in range(args+1):
                for j in range(args+1):
                    w1, w2, w3 = utils.generate_weights(args, i, j)
                    
                    _, newrow = utils.process_weighted_model(model, data, w1 ,w2, w3)
                    
                    csv_data.append(newrow)
                
                    if i == 0:
                        break
            
            
            
        elif all(isinstance(arg, float) for arg in args):
            print(f"Proccessing the optimal ILP solution with weights: {args}")
                        
            w1, w2, w3 = args
            
            concrete, newrow = utils.process_weighted_model(model, data, w1 ,w2, w3)
                  
            csv_data.append(newrow)
            
            concrete.pprint()
        
            
        else:
            sys.exit(f'The algorithm parameters must be a number of subdivisions s or three weights w1,w2,w3.')
        
        
        # Write data in a CSV file.
        filename = f"C:/Users/X1502/eclipse-workspace/git/M2I-TFM-Adriana/output/output.csv"
        
        if os.path.exists(filename):
            os.remove(filename)
                
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
            print("CSV filed correctly created.")
    

    




