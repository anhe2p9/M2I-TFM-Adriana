import os
import re
import ast
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import zipfile
import shutil

# --- CONFIGURACIÓN ---
# Calculamos la ruta absoluta de la carpeta donde está este script (refactor_java)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construimos la ruta exacta al JAR usando esa ruta base
JAR_PATH = os.path.join(SCRIPT_DIR, "target", "refactor-app-1.0-SNAPSHOT-jar-with-dependencies.jar")


def parse_solution_tuple(val):
    """Convierte el string '(3, 3, 7)' en una tupla real de Python."""
    try:
        return ast.literal_eval(val)
    except Exception:
        return ()


def extract_flat_offsets(offset_str):
    """Extrae los pares [start, end] directamente de la columna offsets."""
    try:
        data = ast.literal_eval(str(offset_str))
        flat_offsets = []

        # Procesa directamente el formato [[6327, 6856], [7227, 8777]]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list) and len(item) == 2:
                    flat_offsets.append(f"{item[0]}-{item[1]}")

        return flat_offsets
    except Exception as e:
        print(f"Error parseando offsets: {e}")
        return []


def validate_with_cache(method_name, offsets, cache_path):
    """Lógica para comprobar los offsets con el refactoring cache."""
    # TODO: Leer el cache_path y validar contra inicio y fin.
    return True


def get_sorting_indices(folder_objs_str, user_priority):
    """
    Calcula dinámicamente qué índices de la tupla usar según la carpeta y las preferencias.
    """
    # 1. Extraer los objetivos del nombre de la carpeta y normalizar 'ex' a 'extractions'
    folder_objs = [obj.lower() for obj in folder_objs_str.split('-')]
    folder_objs = ['extractions' if obj == 'ex' else obj for obj in folder_objs]

    # 2. Normalizar las prioridades del usuario
    user_priority = [p.lower() for p in user_priority]
    user_priority = ['extractions' if p == 'ex' else p for p in user_priority]

    # 3. Buscar los índices correspondientes
    indices = []
    for p in user_priority:
        if p in folder_objs:
            indices.append(folder_objs.index(p))
        else:
            print(f"⚠️ Aviso: El objetivo '{p}' no está en la carpeta ({folder_objs_str}). Se ignorará.")

    # Si por algún error no hay índices válidos, ordenamos tal cual vienen en la tupla
    if not indices:
        indices = list(range(len(folder_objs)))

    return indices


def process_results(results_base_dir, target_algo, user_priority, target_class=None):
    """Recorre las carpetas, extrae el óptimo lexicográfico y agrupa offsets."""
    class_offsets_map = {}
    base_path = Path(results_base_dir)

    # Expresión regular: Algorithm_objectives_classPath.java_methodName
    folder_pattern = re.compile(r"^(?P<algo>[^_]+)_(?P<objs>[^_]+)_(?P<classpath>.*\.java)_(?P<method>.+)$")

    print(f"Buscando soluciones en: {results_base_dir}")
    print(f"Algoritmo objetivo: {target_algo}")
    print(f"Prioridad lexicográfica seleccionada: {user_priority}")

    for root, dirs, files in os.walk(base_path):
        current_folder = os.path.basename(root)
        match = folder_pattern.match(current_folder)

        if match:
            algo = match.group("algo")
            objs_str = match.group("objs")
            classpath = match.group("classpath")
            method = match.group("method")

            if algo != target_algo:
                continue
            if target_class and not classpath.endswith(target_class):
                continue

            csv_path = os.path.join(root, f"{method}_complete_data.csv")

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['parsed_solution'] = df['solution'].apply(parse_solution_tuple)

                # Descartar filas que no hayan parseado correctamente una tupla del tamaño esperado
                df = df[df['parsed_solution'].apply(lambda x: len(x) > 0)]
                if df.empty:
                    continue

                # --- MAGIA DINÁMICA DE ORDENACIÓN ---
                # Averiguamos los índices a ordenar para ESTA carpeta específica
                sort_indices = get_sorting_indices(objs_str, user_priority)

                # Creamos la lambda que extrae esos índices. Ej: si sort_indices es [2, 0], saca (x[2], x[0])
                df_sorted = df.sort_values(
                    by='parsed_solution',
                    key=lambda col: col.map(lambda x: tuple(x[i] for i in sort_indices if i < len(x)))
                )

                best_row = df_sorted.iloc[0]
                best_solution = best_row['parsed_solution']
                extracted_offsets = extract_flat_offsets(best_row['offsets'])

                cache_path = os.path.join(root, "refactoring_cache.json")
                if validate_with_cache(method, extracted_offsets, cache_path):

                    real_class_path = classpath.replace(".", "/")
                    real_class_path = real_class_path[:-5] + ".java"

                    if real_class_path not in class_offsets_map:
                        class_offsets_map[real_class_path] = []

                    class_offsets_map[real_class_path].extend(extracted_offsets)
                    print(f"  [+] Método '{method}'. Objetivos carpeta: {objs_str}")
                    print(f"      Óptimo lexicográfico (Tupla original): {best_solution}")

    return class_offsets_map


def apply_refactorings_to_classes(class_offsets_map, project_root):
    """Llama a JavaParser por cada clase con todos sus offsets."""
    for class_rel_path, offsets in class_offsets_map.items():
        if not offsets:
            print(f"No offsets available: {offsets}.")
            continue

        full_class_path = os.path.join(project_root, class_rel_path)

        if not os.path.exists(full_class_path):
            print(f"⚠️ No se encontró el archivo: {full_class_path}")
            continue

        print(f"\n🚀 Refactorizando clase: {os.path.basename(full_class_path)}")

        cmd = ['java', '-jar', JAR_PATH, full_class_path] + offsets
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("   ✅ Refactorización aplicada con éxito.")
        else:
            print("   ❌ Error en JavaParser:\n", result.stderr)


def main():
    parser = argparse.ArgumentParser(description="Refactorización AST por Lotes desde Pareto")

    parser.add_argument("--results-dir", required=True, help="Carpeta base de resultados (ej: MOILP_results_ASE)")
    parser.add_argument("--project-root", required=True, help="Ruta al proyecto Java (puede ser un .zip o una carpeta)")
    parser.add_argument("--algorithm", required=True, choices=["EpsilonConstraintAlgorithm", "HybridMethodAlgorithm"],
                        help="Algoritmo a procesar")

    parser.add_argument("--priority", nargs="+", default=["loc", "extractions", "cc"],
                        help="Orden de objetivos para el óptimo lexicográfico. (Ej: --priority loc cc extractions)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-class", help="Archivo de clase a refactorizar")
    group.add_argument("--all", action="store_true", help="Procesar todo el proyecto")

    args = parser.parse_args()

    # --- GESTIÓN AUTOMÁTICA DEL .ZIP ---
    project_root = args.project_root
    project_name = os.path.basename(os.path.normpath(project_root))

    if project_name.endswith(".zip"):
        clean_name = project_name[:-4]
        # Crear una ruta para extraerlo en el mismo sitio donde está el zip
        extraction_path = os.path.join(os.path.dirname(project_root), f"{clean_name}_unzipped")

        # --- LÓGICA DE LIMPIEZA ---
        # Si la carpeta temporal de extracción ya existe, la borramos para empezar de cero
        if os.path.exists(extraction_path):
            print(f"\n🧹 Borrando versión anterior para empezar limpio: {extraction_path}")
            shutil.rmtree(extraction_path)

        # Extraemos el zip limpio
        print(f"\n📦 Detectado archivo .zip. Descomprimiendo {project_name} limpio...")
        with zipfile.ZipFile(project_root, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)

        # --- NUEVA LÓGICA DE RENOMBRADO AUTOMÁTICO ---
        # Muchos zips contienen una única carpeta raíz dentro. Comprobamos si es el caso.
        inner_contents = os.listdir(extraction_path)
        if len(inner_contents) == 1 and os.path.isdir(os.path.join(extraction_path, inner_contents[0])):
            original_inner_folder = inner_contents[0]
            original_inner_path = os.path.join(extraction_path, original_inner_folder)

            # Creamos la nueva ruta añadiendo '-refactored' al nombre de la carpeta
            new_inner_folder = f"{original_inner_folder}-refactored"
            new_inner_path = os.path.join(extraction_path, new_inner_folder)

            # Renombramos la carpeta físicamente en el disco duro
            print(f"✨ Renombrando carpeta interna a: {new_inner_folder}")
            os.rename(original_inner_path, new_inner_path)

            # Establecemos la raíz del proyecto para que Java trabaje sobre la nueva carpeta renombrada
            project_root = new_inner_path
        else:
            project_root = extraction_path

        project_name = clean_name  # Usamos el nombre limpio para buscar en los resultados

    # --- BÚSQUEDA DE RESULTADOS ---
    target_results_dir = os.path.join(args.results_dir, project_name, f"{project_name}-3-objectives-results")

    if not os.path.exists(target_results_dir):
        print(f"\n⚠️ Error: No se encontró la carpeta de 3 objetivos en:\n{target_results_dir}")
        return

    print(f"\n📁 Carpeta objetivo localizada automáticamente: \n{target_results_dir}")

    # Llamamos a process_results usando la nueva ruta
    class_map = process_results(target_results_dir, args.algorithm, args.priority, args.target_class)

    if not class_map:
        print("\n⚠️ No se encontraron resultados que aplicar.")
        return

    print("\n--- INICIANDO REFACTORIZACIÓN EN LOTE (AST) ---")
    # Pasamos project_root, que ahora apuntará a la carpeta descomprimida (o a la original si no era zip)
    print(f"Class map: {class_map}.")
    apply_refactorings_to_classes(class_map, project_root)


if __name__ == "__main__":
    main()