import os
import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV

# Ruta raíz donde están todas las carpetas de proyectos
# ROOT_DIR = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/hybrid_method_docker_output"


def generate_statistics(input_path: str, output_path: str):
    # Inicializar listas
    resultados = []
    archivos_invalidos = []

    # Recorrer todos los proyectos
    for proyecto in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, proyecto)
        if not os.path.isdir(ruta_proyecto):
            continue

        for clase_metodo in os.listdir(ruta_proyecto):
            ruta_clase = os.path.join(ruta_proyecto, clase_metodo)
            if not os.path.isdir(ruta_clase):
                continue

            results_file = next((f for f in os.listdir(ruta_clase) if f.endswith("_results.csv")), None)
            nadir_file = next((f for f in os.listdir(ruta_clase) if f.endswith("_nadir.csv")), None)
            if results_file is None or nadir_file is None:
                continue

            results_path = os.path.join(ruta_clase, results_file)

            nadir_path = os.path.join(ruta_clase, nadir_file)
            df_nadir = pd.read_csv(nadir_path)
            nadir = df_nadir.iloc[0].values.astype(float)

            try:
                df = pd.read_csv(results_path)

                # Validar que hay datos y al menos 3 columnas numéricas
                if df.shape[0] == 0:
                    raise ValueError("Archivo vacío")

                columnas_numericas = df.select_dtypes(include=[np.number])
                if columnas_numericas.shape[1] < 3:
                    raise ValueError("Menos de 3 columnas numéricas")

                # Seleccionar primeras 3 columnas numéricas
                objetivos = columnas_numericas.iloc[:, :3].values
                nombres_objetivos = columnas_numericas.columns[:3]

                if objetivos.size == 0:
                    raise ValueError("Sin datos numéricos")

                # Punto de referencia y cálculo de hipervolumen
                hv = HV(ref_point=nadir)
                hipervolumen = hv.do(objetivos)

                # Cálculo del hipervolumen de la caja total
                hv_max = hv.do(np.array([0,0,0]))

                hv_normalized = hipervolumen / hv_max


                # Dividir clase_metodo en clase y método
                if "_" in clase_metodo:
                    clase, metodo = clase_metodo.rsplit("_", 1)
                else:
                    clase, metodo = clase_metodo, ""

                # Estadísticas
                medias = np.round(np.mean(objetivos, axis=0), 2)
                desvios = np.round(np.std(objetivos, axis=0), 2)
                medianas = np.median(objetivos, axis=0)
                iqr = np.percentile(objetivos, 75, axis=0) - np.percentile(objetivos, 25, axis=0)

                resultados.append({
                    "project": proyecto,
                    "class": clase,
                    "method": metodo,
                    "hypervolume": hipervolumen,
                    "normalized_hypervolume": np.round(hv_normalized,2),
                    f"avg_{nombres_objetivos[0]}": medias[0],
                    f"std_{nombres_objetivos[0]}": desvios[0],
                    f"median_{nombres_objetivos[0]}": medianas[0],
                    f"iqr_{nombres_objetivos[0]}": iqr[0],
                    f"avg_{nombres_objetivos[1]}": medias[1],
                    f"std_{nombres_objetivos[1]}": desvios[1],
                    f"median_{nombres_objetivos[1]}": medianas[1],
                    f"iqr_{nombres_objetivos[1]}": iqr[1],
                    f"avg_{nombres_objetivos[2]}": medias[2],
                    f"std_{nombres_objetivos[2]}": desvios[2],
                    f"median_{nombres_objetivos[2]}": medianas[2],
                    f"iqr_{nombres_objetivos[2]}": iqr[2],
                })


            except Exception as e:
                archivos_invalidos.append({
                    "proyecto": proyecto,
                    "clase_metodo": clase_metodo,
                    "archivo": results_path,
                    "error": str(e)
                })

    # Guardar resumen principal
    df_resultados = pd.DataFrame(resultados)
    ruta_salida = os.path.join(output_path, "hipervolumen_summary.csv")
    df_resultados.to_csv(ruta_salida, index=False)
    print(f"\n✅ Resumen guardado en: {ruta_salida}")

    # Guardar archivos inválidos (opcional)
    if archivos_invalidos:
        df_invalidos = pd.DataFrame(archivos_invalidos)
        ruta_invalidos = os.path.join(output_path, "archivos_invalidos.csv")
        df_invalidos.to_csv(ruta_invalidos, index=False)
        print(f"⚠️  Algunos archivos fueron descartados por errores. Detalles en: {ruta_invalidos}")
