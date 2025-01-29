import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import csv
import os
import sys
import argparse
from typing import Any
from pathlib import Path


from ILP_CC_reducer.operations.ILP_engine import ILPEngine
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES
    
# code_filepath: str, model: pyo.AbstractModel, algorithm: str = None, subdivisions: int = None

def main(instance_folder: Path, algorithm: str, tau: int=15, subdivisions: int=6, weights: str=None):
    
    model_engine = ILPEngine()
    
    # Proccess weights
    if weights:
        weights = list(map(int, args.weights.split(',')))
        if len(weights) != 3:
            sys.exit("Weights parameter w1,w2,w3 must be exactly three weights separated by comma (',').")
            
            
    
            
            
    model_engine.load_concrete(instance_folder, tau)
    
    
    # Procesar los datos
    # Pasar los datos al algoritmo que sea
    # Obtener el resultado y escribirlo en un fichero CSV
    
    
    

    
def resultsWriter(result_data: list[list[Any]]):
        
    # print(result_data)
    
    # Escribir datos en un archivo CSV
    with open("C:/Users/X1502/eclipse-workspace/git/M2I-TFM-Adriana/results_new.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(result_data)
    
    print("Archivo CSV creado correctamente.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refactoring engine. Given an abstract model m, a model instance a, an algorithm a and optionally a determined number of subdivisions s or three weights w, it applies the correspondent algorithm to find the optimal solutions of the model instance.')
    parser.add_argument('-i', '--instance', dest='model_instance', type=str, required=True, help='Model instance to be optimized (folder with the three data files in CSV format).')
    parser.add_argument('-a', '--algorithm', dest='ilp_algorithm', type=str, required=True, help=f'Algorithm to be applied to the model instance {[a for a in ALGORITHMS_NAMES]}.')
    parser.add_argument('-t', '--tau', dest='threshold', type=str, required=False, help=f'Threshold (tau) to be reached by the optimization model.')
    parser.add_argument('-s', '--subdivisions', dest='subdivisions', type=str, required=False, help=f'Subdivisions to generate different weights.')
    parser.add_argument('-w', '--weights', dest='weights', type=float, nargs=3, required=False, help=f'Weights assigned for weighted sum in the case of a specific combination of weights. Three weights w1,w2,w3 separated with comma (",").')
    
    args = parser.parse_args()
    
    # -i C:/Users/X1502/eclipse-workspace/git/M2I-TFM-Adriana/original_code_data/EZInjection_hook -a WeightedSumAlgorithm -t 15 -s 6
    instance_path = Path(args.model_instance)
    if not instance_path.is_dir():
        sys.exit(f'The model instance must be a folder with three CSV files.')
    main(instance_path, args.ilp_algorithm, args.threshold, args.subdivisions, args.weights)