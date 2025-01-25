import csv
import os
import sys
import argparse

# from models.technique import __all__ as TECHNIQUE_NAMES
    


def main(fm_filepath: str, instance_identifier: str = None, refactoring_name: str = None):
    # Get feature model name
    path, filename = os.path.split(fm_filepath)
    # filename = ''.join(filename.split('.')[:-1])

    # Load the feature model
    print(f'Reading code data from {fm_filepath}...')

    
    
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
    parser = argparse.ArgumentParser(description='Refactoring engine. Given a FM and optionally a feature/constraint i or a refactoring r, apply the appropriate refactoring (if any) to i, and generate the new FM. If not instance or refactoring is provided, all available refactorings are applied.')
    parser.add_argument('-fm', '--featuremodel', dest='feature_model', type=str, required=True, help='Input feature model in UVL format.')
    parser.add_argument('-i', '--instance', dest='instance', type=str, required=False, help='Instance to be refactored (name of the feature or number of constraint [0..n-1]).')
    parser.add_argument('-r', '--refactoring', dest='refactoring', type=str, required=False, help=f'Refactoring to be applied to all instances {[r for r in REFACTORINGS_NAMES]}.')
    args = parser.parse_args()

    if not args.feature_model.endswith('.uvl'):
        sys.exit(f'The FM must be in UVL format (.uvl).')
    main(args.feature_model, args.instance, args.refactoring)