import os
import subprocess
import sys

ROOT_DIR = sys.argv[1]

PYTHON_CMD = "python"
MAIN_SCRIPT = "main.py"

N_OBJECTIVES = 3
ALGORITHM = sys.argv[2]
TAU = 15
TIME_LIMIT = 120

FLAGS = ["--plot", "--3dPF", "--relHV"]

if not os.path.isdir(ROOT_DIR):
    raise ValueError(f"Root folder not found: {ROOT_DIR}")

for project in os.listdir(ROOT_DIR):
    project_path = os.path.join(ROOT_DIR, project)
    if not os.path.isdir(project_path):
        continue

    print(f"Proyecto: {project}")

    for cls in os.listdir(project_path):
        class_path = os.path.join(project_path, cls)
        if not os.path.isdir(class_path):
            continue

        print(f"  Clase: {cls}")

        for method in os.listdir(class_path):
            method_path = os.path.join(class_path, method)
            if not os.path.isdir(method_path):
                continue

            print(f"    Método: {method}")
            print(f"    Ejecutando instancia en: {method_path}")

            cmd = [
                PYTHON_CMD, MAIN_SCRIPT,
                "-n", str(N_OBJECTIVES),
                "-i", method_path,
                "-a", ALGORITHM,
                "-t", str(TAU),
                "-tl", str(TIME_LIMIT),
                *FLAGS
            ]

            subprocess.run(cmd, check=True)
