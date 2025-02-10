import os
import sys
import argparse
import configparser
from pathlib import Path


from ILP_CC_reducer.operations.ILP_engine import ILPEngine
from ILP_CC_reducer.algorithms import __all__ as ALGORITHMS_NAMES

from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel 
    
# code_filepath: str, model: pyo.AbstractModel, algorithm: str = None, subdivisions: int = None

def main(alg_name: str, instance_folder: Path, tau: int=15, subdivisions=None, weights=None, second_obj=None):

    print(f"Second objective: {second_obj}")    
    model_engine = ILPEngine()
    model = MultiobjectiveILPmodel()
    
    # Process ilp model
    ilp_model = model.define_model_without_obj()
    
    # Process algorithm
    algorithm = model_engine.get_algorithm_from_name(alg_name)
    
    # Process instance
    instance = model_engine.load_concrete(instance_folder)
    
    model_engine.apply_algorithm(algorithm, ilp_model, instance, tau, subdivisions, weights, second_obj)

    # print(result)
    

PROPERTIES_FILE = "properties.ini"

def delete_ini(path):
    if os.path.exists(path):
        os.remove(path)

def load_config(file=PROPERTIES_FILE):
    """Loads configuration from a file .ini if it exists."""

    config = configparser.ConfigParser()
    config.read(file)

    parameters = {}

    if "Properties" in config:
        section = config["Properties"]
        parameters["model_instance"] = section["model_instance"]
        if "ilp_algorithm" in section:
            parameters["ilp_algorithm"] = section["ilp_algorithm"]
        if "threshold" in section:
            parameters["threshold"] = section.getint("threshold")
        if "subdivisions" in section:
            parameters["subdivisions"] = section.getint("subdivisions")
        if "weights" in section:
            parameters["weights"] = section["weights"]
        if "second_obj" in section:
            parameters["second_obj"] = section["second_obj"]

    return parameters





def save_config(parameters, file=PROPERTIES_FILE):
    """Saves properties in a .ini file"""
    
    
    config = configparser.ConfigParser()
    # Convert lists/tuples into string before saving them
    config["Properties"] = {
        key: ",".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
        for key, value in parameters.items()
    }

    with open(file, "w") as f:
        config.write(f)

    print(f"Properties saved in {file}")
    
    
    
    

def obtain_arguments():
    """Defines arguments from command line and parse them."""

    parser = argparse.ArgumentParser(description='ILP model engine. Given an abstract model m, a model instance a, an algorithm a and optionally a determined number of subdivisions s or three weights w, it applies the correspondent algorithm to find the optimal solutions of the model instance. One can also give as input a properties file path.')
    parser.add_argument('-f', '--file', dest='properties_file', type=str, default=None, help=f'Properties file name in case one want to give every parameter from a .ini file.')
    parser.add_argument('-i', '--instance', dest='model_instance', type=str, default=None, help='Model instance to be optimized (name of the folder with the three data files in CSV format).')
    parser.add_argument('-a', '--algorithm', dest='ilp_algorithm', type=str, default=None, help=f'Algorithm to be applied to the model instance {[a for a in ALGORITHMS_NAMES]}.')
    parser.add_argument('-t', '--tau', dest='threshold', type=int, default=None, help=f'Threshold (tau) to be reached by the optimization model.')
    parser.add_argument('-s', '--subdivisions', dest='subdivisions', type=int, default=None, help=f'Number of subdivisions to generate different weights.')
    parser.add_argument('-w', '--weights', dest='weights', type=str, default=None, help=f'Weights assigned for weighted sum in the case of a specific combination of weights. Three weights w1,w2,w3 separated by comma (",").')
    parser.add_argument('-o', '--secondobj', dest='second_obj', type=str, default=None, help=f'Second objective to minimize in the case of a two objective ILP.')
    parser.add_argument('--save', action='store_true', help='Save properties in a .ini file')

    
    args = parser.parse_args()
    parameters = vars(args)
    

    return parameters

    








if __name__ == '__main__':
    
    # Obtain arguments from command-line
    args = obtain_arguments()
    
    # Load properties from file if it exists
    config = {}
    if args['properties_file']:
        properties_file_path = Path(args['properties_file'])
        print(f"PROPERTIES FILE PATH: {properties_file_path}")
        if not properties_file_path.is_file():
            sys.exit(f'The model instance must be a .ini file.')
        config = load_config(properties_file_path)
    
    
    
    model_instance = args['model_instance'] if args['model_instance'] else config.get('model_instance')
    ilp_algorithm = args['ilp_algorithm'] if args['ilp_algorithm'] else config.get('ilp_algorithm')
    threshold = args['threshold'] if args['threshold'] else config.get('threshold')
    subdivisions = args['subdivisions'] if args['subdivisions'] else config.get('subdivisions')
    weights = args['weights'] if args['weights'] else config.get('weights')
    second_obj = args['second_obj'] if args['second_obj'] else config.get('second_obj')
    
    # Overwrite .ini file values with commandline values if it exists
    for key, value in args.items():
        if value:  # Solo actualizar si el usuario lo pasó por línea de comandos
            config[key] = value
    
    # Save file if there is '--save'
    if args["save"]:
        save_config(config)

    # Show final properties used
    print("Final configuration:")
    for key, value in config.items():
        print(f"   · {key} = {value}")  
    
    
    
    instance_path = Path(model_instance)
    if not instance_path.is_dir():
        sys.exit(f'The model instance must be a folder with three CSV files.')
        
    # Turn "x,y,z" into (float,float,float) if --weights is a parameter in command line
    if weights:
        weights = tuple(map(float, weights.split(",")))
        
    
    main(ilp_algorithm, instance_path, threshold, subdivisions, weights, second_obj)

        
    # if config['ilp_algorithm'] == 'WeightedSumAlgorithm' or config['ilp_algorithm'] == 'WeightedSumAlgorithm2obj':
    #     if config.get('subdivisions'):
    #         main(instance_path, ilp_algorithm, threshold, subdivisions)
    #     elif config.get('weights'):
    #         main(instance_path, ilp_algorithm, threshold, weights)
    #     else:
    #         sys.exit(f'The Weighted Sum Algorithm parameters must be a number of subdivisions s or three weights w1,w2,w3.')
    # else:
    #     main(instance_path, ilp_algorithm, threshold, subdivisions, weights, second_obj)
    #
    #

    
    # PARÁMETROS DE PRUEBA
    # -i C:/Users/X1502/eclipse-workspace/git/M2I-TFM-Adriana/original_code_data/EZInjection_hook -a WeightedSumAlgorithm -t 15 -s 6
    
    
    
    
    
    
    
    
    
    
    