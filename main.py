import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import csv
import os
import sys
import argparse

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES
    


def main(code_filepath: str, model: pyo.AbstractModel, algorithm: str = None, subdivisions: int = None):
    
    
    # Procesar los datos
    # Pasar los datos al algoritmo que sea
    # Obtener el resultado y escribirlo en un fichero CSV
    
    
    
    # Get data name
    code_path, code_filename = os.path.split(code_filepath)
    
    S_filename = ''.join(code_filename.split('.')[:-1])

    # Load the code data
    print(f'Reading code data from {S_filepath}...')
    
        
    data = dp.DataPortal()
    data.load(filename=S_filename, index=model.S, param=(model.loc, model.nmcc))
    data.load(filename=N_filename, index=model.N, param=model.ccr)
    data.load(filename=C_filename, index=model.C, param=())
    
    
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
    data = pyo.DataPortal()
    if data_files["parameters"]:
        data.load(filename=data_files["parameters"], param=modelo.<nombre_del_parametro>)
    if data_files["sets"]:
        data.load(filename=data_files["sets"], set=modelo.<nombre_del_conjunto>)
    if data_files["data"]:
        data.load(filename=data_files["data"], param=modelo.<otros_parametros>)
    
    # Crear el modelo concreto con los datos cargados
    instancia = modelo.create_instance(data)
    return instancia
    
    
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
    parser.add_argument('-m', '--model', dest='abstract_model', type=str, required=True, help='Abstract ILP model to be used (ILPmodelRsain for one objective and multiobILPmodel for multiobjective problem).')
    parser.add_argument('-i', '--instance', dest='model_instance', type=str, required=True, help='Model instance to be optimized (folder with the three data files in CSV format).')
    parser.add_argument('-t', '--tau', dest='threshold', type=str, required=True, help=f'Threshold (tau) to be reached by the optimization model.')
    parser.add_argument('-a', '--algorithm', dest='ilp_algorithm', type=str, required=False, help=f'Algorithm to be applied to the model instance {[a for a in ALGORITHMS_NAMES]}.')
    parser.add_argument('-s', '--subdivisions', dest='subdivisions', type=str, required=False, help=f'Subdivisions to generate different weights.')
    parser.add_argument('-w', '--weights', dest='ilp_algorithm', type=float, nargs=3, required=False, help=f'.')
    
    args = parser.parse_args()

    if not args.feature_model.endswith('.uvl'):
        sys.exit(f'The FM must be in UVL format (.uvl).')
    main(args.feature_model, args.instance, args.refactoring)