import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import csv
import os
import sys
import argparse

from ILP_CC_reducer.operations.ILP_engine import ILPEngine
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES
    
# code_filepath: str, model: pyo.AbstractModel, algorithm: str = None, subdivisions: int = None

def main(instance_folder: str, tau: int=15, algorithm: str, subdivisions: int=6, weights: str=None):
    
    model_engine = ILPEngine()
    
    # Model instance file suffixes
    suffixes = ['_sequences.csv', '_nested.csv', '_conflict.csv']    
    
    
    # Process files
    data = dp.DataPortal()
    
    for suffix in suffixes:
        # Search files with each suffix in the given folder
        for file in os.listdir(instance_folder):
            if file.endswith(suffix):
                file_path = os.path.join(instance_folder, file)
                print(f'Procesando archivo: {file_path}')
                
                model_engine.load_concrete(file_path)
    
    
    
    # Proccess weights
    if weights:
        weights = list(map(int, args.weights.split(',')))
        if len(weights) != 3:
            sys.exit("Weights parameter w1,w2,w3 must be exactly three weights separated by comma (',').")
    
    # Procesar los datos
    # Pasar los datos al algoritmo que sea
    # Obtener el resultado y escribirlo en un fichero CSV
    
    
    
    
    
    

    
    
def cargar_instancia_modelo(ruta_carpeta, modelo):
    """
    Carga una instancia para un modelo abstracto de Pyomo desde una carpeta con ficheros CSV.
    
    Args:
        ruta_carpeta (str): Ruta de la carpeta donde se encuentran los archivos CSV.
        modelo (pyo.AbstractModel): El modelo abstracto de Pyomo.
        
    Returns:
        pyo.ConcreteModel: Modelo concreto cargado con los datos.
    """
    # Dictionary to asociate files suffixes to their correspondent role
    file_sufixes = {
        "_sequences.csv": "sequences",
        "_nested.csv": "nested",
        "_conflict.csv": "conflict"
    }

    # Dictionary for storing paths to identified files
    data_files = {key: None for key in file_sufixes.values()}

    # Identificar los data_files según el sufijo
    for datafile in os.listdir(ruta_carpeta):
        for sufijo, clave in file_sufixes.items():
            if datafile.endswith(sufijo):
                data_files[clave] = os.path.join(ruta_carpeta, datafile)

    # Verificar que se encontraron todos los archivos necesarios
    for clave, ruta in data_files.items():
        if ruta is None:
            raise FileNotFoundError(f"No se encontró un archivo con el sufijo correspondiente a '{clave}'.")

    # Cargar los datos desde los archivos detectados
    # data = pyo.DataPortal()
    # if data_files["parameters"]:
    #     data.load(filename=data_files["parameters"], param=modelo.<nombre_del_parametro>)
    # if data_files["sets"]:
    #     data.load(filename=data_files["sets"], set=modelo.<nombre_del_conjunto>)
    # if data_files["data"]:
    #     data.load(filename=data_files["data"], param=modelo.<otros_parametros>)
    #

    # # Crear el modelo concreto con los datos cargados
    # instancia = modelo.create_instance(data)
    # return instancia
    #

    
def resultsWriter():
    # Nombre del archivo CSV
    filename = "output.csv"
    
    # Datos a escribir en el archivo
    data = [
        ["Nombre", "Edad", "Ciudad"],
        ["Ana", 28, "Madrid"],
        ["Carlos", 34, "Barcelona"],
        ["Lucía", 25, "Valencia"]
    ]
    
    try:
        # Abrir el archivo en modo escritura
        with open(filename, mode="w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
    
            # Escribir las filas en el archivo
            for row in data:
                writer.writerow(row)
    
        print(f"Archivo {filename} escrito con éxito.")
    except Exception as e:
        print(f"Error al escribir el archivo: {e}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refactoring engine. Given an abstract model m, a model instance a, an algorithm a and optionally a determined number of subdivisions s or three weights w, it applies the correspondent algorithm to find the optimal solutions of the model instance.')
    parser.add_argument('-i', '--instance', dest='model_instance', type=str, required=True, help='Model instance to be optimized (folder with the three data files in CSV format).')
    parser.add_argument('-t', '--tau', dest='threshold', type=str, required=True, help=f'Threshold (tau) to be reached by the optimization model.')
    parser.add_argument('-a', '--algorithm', dest='ilp_algorithm', type=str, required=True, help=f'Algorithm to be applied to the model instance {[a for a in ALGORITHMS_NAMES]}.')
    parser.add_argument('-s', '--subdivisions', dest='subdivisions', type=str, required=False, help=f'Subdivisions to generate different weights.')
    parser.add_argument('-w', '--weights', dest='ilp_algorithm', type=float, nargs=3, required=False, help=f'Weights assigned for weighted sum in the case of a specific combination of weights. Three weights w1,w2,w3 separated with comma (",").')
    
    args = parser.parse_args()

    if not args.instance.isdir():
        sys.exit(f'The model instance must be a folder with three CSV files.')
    main(args.instance, args.tau, args.algorithm, args.subdivisions, args.weights)