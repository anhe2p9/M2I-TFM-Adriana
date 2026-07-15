import os
import re
import ast
import glob
import sys
import json
import time
import argparse
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
import zipfile
import shutil

# --- CONFIGURACIÓN DE ECLIPSE JDT LS ---
# Intenta leer la variable de entorno JDTLS_HOME
JDTLS_HOME = os.getenv("JDTLS_HOME")

# Si no existe (devuelve None), detenemos el script con un error claro
if not JDTLS_HOME:
    print("❌ Error: La variable de entorno 'JDTLS_HOME' no está configurada.")
    print("Por favor, configúrala apuntando a la carpeta de Eclipse JDT LS.")
    sys.exit(1)


# =====================================================================
# CLIENTE LSP NATIVO PARA ECLIPSE JDT LS (VERSIÓN MEJORADA Y ROBUSTA)
# =====================================================================
class JDTLSClient:
    def __init__(self, jdtls_home, workspace_dir, project_path):
        # Convertimos todas las rutas a absolutas con barras '/' para evitar conflictos en Windows
        self.jdtls_home = os.path.abspath(jdtls_home).replace("\\", "/")
        self.workspace_dir = os.path.abspath(workspace_dir).replace("\\", "/")
        self.project_path = os.path.abspath(project_path).replace("\\", "/")
        self.proc = None
        self.request_id = 1
        self.file_versions = {}

    def start(self):
        import sys  # <--- IMPORTANTE: Importado al principio para evitar el UnboundLocalError

        plugins_dir = f"{self.jdtls_home}/plugins"
        launchers = glob.glob(os.path.join(plugins_dir, "org.eclipse.equinox.launcher_*.jar"))
        if not launchers:
            raise FileNotFoundError("❌ No se encontró el JAR de equinox launcher en jdtls/plugins. Revisa JDTLS_HOME.")
        launcher_jar = os.path.abspath(launchers[0]).replace("\\", "/")

        if sys.platform.startswith("win"):
            config_dir = "config_win"
        elif sys.platform.startswith("darwin"):
            config_dir = "config_mac"
        else:
            config_dir = "config_linux"
        config_path = os.path.abspath(os.path.join(self.jdtls_home, config_dir)).replace("\\", "/")

        cmd = [
            "java",
            "-Declipse.application=org.eclipse.jdt.ls.core.id1",
            "-Dosgi.bundles.defaultStartLevel=4",
            "-Declipse.product=org.eclipse.jdt.ls.core.product",
            "-Dlog.level=ALL",
            "-Xmx1G",
            "--add-modules=ALL-SYSTEM",
            "--add-opens", "java.base/java.util=ALL-UNNAMED",
            "--add-opens", "java.base/java.lang=ALL-UNNAMED",
            "-jar", launcher_jar,
            # --- ARGUMENTOS DE ECLIPSE (Siempre deben ir después de -jar) ---
            "-configuration", config_path,
            "-data", self.workspace_dir,
            "-noconsole"
        ]

        print("\n🚀 Intentando iniciar Eclipse JDT LS con el siguiente comando:")
        print(" ".join([f'"{arg}"' if " " in arg else arg for arg in cmd]))
        sys.stdout.flush()

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(2.0)
        poll = self.proc.poll()
        if poll is not None:
            stdout_output = self.proc.stdout.read().decode('utf-8', errors='ignore') if self.proc.stdout else ""
            stderr_output = self.proc.stderr.read().decode('utf-8', errors='ignore') if self.proc.stderr else ""

            print("\n❌ Error crítico: El proceso de Eclipse JDT LS se cerró inmediatamente al arrancar.")
            print("--- LOG DE ERROR DE JAVA (STDERR) ---")
            print(stderr_output if stderr_output.strip() else "No hay salida en STDERR.")
            print("-------------------------------------")
            print("--- LOG DE SALIDA DE JAVA (STDOUT) ---")
            print(stdout_output if stdout_output.strip() else "No hay salida en STDOUT.")
            print("--------------------------------------")
            sys.stdout.flush()
            raise RuntimeError("No se pudo iniciar Eclipse JDT LS debido a un fallo en la JVM.")

    def send(self, method, params=None, is_notification=False):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not is_notification:
            payload["id"] = self.request_id
            self.request_id += 1

        body = json.dumps(payload)
        header = f"Content-Length: {len(body)}\r\n\r\n"

        try:
            self.proc.stdin.write((header + body).encode('utf-8'))
            self.proc.stdin.flush()
        except OSError as e:
            # Si el canal está roto, informamos de la muerte del servidor
            self._handle_server_crash()
            raise e
        return payload.get("id")

    def _handle_server_crash(self):
        if self.proc:
            stderr_output = self.proc.stderr.read().decode('utf-8', errors='ignore')
            print("\n❌ El servidor Eclipse JDT LS ha muerto inesperadamente.")
            print("--- LOG DE ERROR DE JAVA (STDERR) ---")
            print(stderr_output)
            print("-------------------------------------")

    def read_message(self):
        if self.proc.poll() is not None:
            self._handle_server_crash()
            raise RuntimeError("El proceso JDT LS ha terminado de manera abrupta.")

        content_length = None
        while True:
            line = self.proc.stdout.readline().decode('utf-8').strip()
            if not line:
                if self.proc.poll() is not None:
                    self._handle_server_crash()
                    raise RuntimeError("El proceso JDT LS ha terminado mientras se leía del buffer.")
                break
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":")[1].strip())
        if content_length is None:
            return None
        body = self.proc.stdout.read(content_length).decode('utf-8')
        return json.loads(body)

    def wait_for_response(self, req_id, timeout=15):
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.read_message()
            if not msg:
                continue
            if msg.get("method") == "workspace/applyEdit":
                self.apply_workspace_edit(msg["params"]["edit"])
                self.send_response(msg["id"], {"applied": True})
            elif msg.get("id") == req_id:
                return msg
        raise TimeoutError(f"⏳ Timeout esperando respuesta del servidor JDT LS para ID {req_id}")

    def send_response(self, req_id, result):
        payload = {"jsonrpc": "2.0", "id": req_id, "result": result}
        body = json.dumps(payload)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        try:
            self.proc.stdin.write((header + body).encode('utf-8'))
            self.proc.stdin.flush()
        except OSError:
            self._handle_server_crash()

    def initialize(self):
        project_uri = f"file:///{self.project_path}"
        req_id = self.send("initialize", {
            "processId": os.getpid(),
            "rootUri": project_uri,
            "capabilities": {
                "workspace": {"applyEdit": True},
                "textDocument": {
                    "codeAction": {
                        "codeActionLiteralSupport": {
                            "codeActionKind": {"valueSet": ["refactor.extract"]}
                        }
                    }
                }
            }
        })
        self.wait_for_response(req_id)
        self.send("initialized", is_notification=True)
        print("☕ Eclipse JDT LS inicializado y compilando el proyecto...")
        time.sleep(3)

    def open_file(self, file_path):
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        uri = f"file:///{abs_path}"
        with open(file_path, "r", encoding="utf-8", newline='') as f:
            content = f.read()
        self.file_versions[uri] = self.file_versions.get(uri, 0) + 1
        self.send("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "java",
                "version": self.file_versions[uri],
                "text": content
            }
        }, is_notification=True)

    def close_file(self, file_path):
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        uri = f"file:///{abs_path}"
        self.send("textDocument/didClose", {"textDocument": {"uri": uri}}, is_notification=True)

    def request_extract_method(self, file_path, start_pos, end_pos):
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        uri = f"file:///{abs_path}"

        req_id = self.send("textDocument/codeAction", {
            "textDocument": {"uri": uri},
            "range": {
                "start": {"line": start_pos["line"], "character": start_pos["character"]},
                "end": {"line": end_pos["line"], "character": end_pos["character"]}
            },
            "context": {"diagnostics": [], "only": ["refactor.extract"]}
        })
        resp = self.wait_for_response(req_id)

        actions = resp.get("result", [])
        extract_action = None
        for action in actions:
            title = action.get("title", "").lower()
            if "extract" in title and "method" in title:
                extract_action = action
                break

        if not extract_action:
            print("⚠️ Eclipse determinó que la extracción de este bloque NO es semánticamente segura. Omitiendo.")
            return False

        if "command" in extract_action:
            cmd = extract_action["command"]
            req_id = self.send("workspace/executeCommand", {
                "command": cmd["command"],
                "arguments": cmd.get("arguments", [])
            })
            self.wait_for_response(req_id)
            return True
        elif "edit" in extract_action:
            self.apply_workspace_edit(extract_action["edit"])
            return True
        return False

    def apply_workspace_edit(self, edit):
        if "changes" in edit:
            for uri, text_edits in edit["changes"].items():
                file_path = uri.replace("file:///", "").replace("/", os.sep)
                self.apply_text_edits(file_path, text_edits)
        elif "documentChanges" in edit:
            for doc_change in edit["documentChanges"]:
                if "textDocument" in doc_change:
                    uri = doc_change["textDocument"]["uri"]
                    file_path = uri.replace("file:///", "").replace("/", os.sep)
                    self.apply_text_edits(file_path, doc_change["edits"])

    def apply_text_edits(self, file_path, edits):
        # Usamos newline='' para no corromper los \r\n de Windows al leer
        with open(file_path, "r", encoding="utf-8", newline='') as f:
            content = f.read()

        # --- CÁLCULO DE DESPLAZAMIENTOS EXACTOS ---
        shifts = []
        for edit in edits:
            start_abs = position_to_offset(content, edit["range"]["start"])
            end_abs = position_to_offset(content, edit["range"]["end"])
            # Cuánto cambia el tamaño localmente = tamaño del texto nuevo - tamaño del texto reemplazado
            delta = len(edit["newText"]) - (end_abs - start_abs)
            shifts.append({"pos": start_abs, "delta": delta})

        # Guardamos los shifts en la instancia para que el bucle principal los recoja
        self.last_shifts = shifts

        sorted_edits = sorted(
            edits,
            key=lambda e: (e["range"]["start"]["line"], e["range"]["start"]["character"]),
            reverse=True
        )

        lines = content.splitlines(keepends=True)
        for edit in sorted_edits:
            start, end = edit["range"]["start"], edit["range"]["end"]
            new_text = edit["newText"]

            s_line, s_char = start["line"], start["character"]
            e_line, e_char = end["line"], end["character"]

            if s_line == e_line:
                lines[s_line] = lines[s_line][:s_char] + new_text + lines[s_line][e_char:]
            else:
                prefix = lines[s_line][:s_char]
                suffix = lines[e_line][e_char:]
                lines[s_line] = prefix + new_text + suffix
                for _ in range(s_line + 1, e_line + 1):
                    lines.pop(s_line + 1)

        with open(file_path, "w", encoding="utf-8", newline='') as f:
            f.write("".join(lines))

    def stop(self):
        if self.proc:
            try:
                # Solo intentamos enviar comandos de apagado si el proceso sigue realmente activo
                if self.proc.poll() is None:
                    self.send("shutdown")
                    self.send("exit", is_notification=True)
            except Exception:
                pass
            finally:
                self.proc.terminate()
                self.proc.wait()


# =====================================================================
# AUXILIARES DE PARSEO Y AJUSTES DE OFFSET
# =====================================================================
def offset_to_position(content, offset):
    """Convierte un offset de carácter absoluto a coordenadas de línea y columna LSP (0-indexed)."""
    before = content[:offset]
    lines = before.split('\n')
    line = len(lines) - 1
    character = len(lines[-1])
    return {"line": line, "character": character}


def position_to_offset(content, pos):
    """Convierte coordenadas de línea y columna LSP a un offset absoluto de caracteres."""
    lines = content.splitlines(keepends=True)
    if pos["line"] >= len(lines):
        return len(content)
    offset = sum(len(lines[i]) for i in range(pos["line"]))
    offset += min(pos["character"], len(lines[pos["line"]]))
    return offset


def adjust_range_with_shifts(remaining_range, shifts):
    """Ajusta dinámicamente un rango basándose en múltiples ediciones locales exactas."""
    start, end = remaining_range
    # Ordenamos los shifts por posición para no desfasar el cálculo en cascada
    for shift in sorted(shifts, key=lambda x: x["pos"]):
        if start > shift["pos"]:
            start += shift["delta"]
        if end > shift["pos"]:
            end += shift["delta"]
    return [start, end]


def parse_solution_tuple(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return ()


def extract_flat_offsets(offset_str):
    try:
        data = ast.literal_eval(str(offset_str))
        flat_offsets = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list) and len(item) == 2:
                    flat_offsets.append(item)  # Guardar como lista numérica [start, end]
        return flat_offsets
    except Exception as e:
        print(f"Error parseando offsets: {e}")
        return []


def get_sorting_indices(folder_objs_str, user_priority):
    folder_objs = [obj.lower() for obj in folder_objs_str.split('-')]
    folder_objs = ['extractions' if obj == 'ex' else obj for obj in folder_objs]
    user_priority = [p.lower() for p in user_priority]
    user_priority = ['extractions' if p == 'ex' else p for p in user_priority]

    indices = []
    for p in user_priority:
        if p in folder_objs:
            indices.append(folder_objs.index(p))
    if not indices:
        indices = list(range(len(folder_objs)))
    return indices


def process_results(results_base_dir, target_algo, user_priority, target_class=None):
    class_offsets_map = {}
    base_path = Path(results_base_dir)
    folder_pattern = re.compile(r"^(?P<algo>[^_]+)_(?P<objs>[^_]+)_(?P<classpath>.*\.java)_(?P<method>.+)$")

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
                df = df[df['parsed_solution'].apply(lambda x: len(x) > 0)]
                if df.empty:
                    continue

                sort_indices = get_sorting_indices(objs_str, user_priority)
                df_sorted = df.sort_values(
                    by='parsed_solution',
                    key=lambda col: col.map(lambda x: tuple(x[i] for i in sort_indices if i < len(x)))
                )

                best_row = df_sorted.iloc[0]
                extracted_offsets = extract_flat_offsets(best_row['offsets'])

                real_class_path = classpath.replace(".", "/")
                real_class_path = real_class_path[:-5] + ".java"

                if real_class_path not in class_offsets_map:
                    class_offsets_map[real_class_path] = []

                class_offsets_map[real_class_path].extend(extracted_offsets)

    return class_offsets_map


# =====================================================================
# MOTOR PRINCIPAL DE REFACTORIZACIÓN CON ECLIPSE JDT LS
# =====================================================================
def apply_refactorings_to_classes(class_offsets_map, project_root):
    """Inicializa Eclipse JDT LS y ejecuta las refactorizaciones dinámicas."""

    # Crear workspace temporal para que Eclipse compile aislado
    temp_ws = tempfile.mkdtemp(prefix="jdtls_ws_")

    print(f"\n🚀 Iniciando Eclipse JDT LS headless...")
    client = JDTLSClient(JDTLS_HOME, temp_ws, project_root)

    try:
        client.start()
        client.initialize()

        for class_rel_path, raw_ranges in class_offsets_map.items():
            if not raw_ranges:
                continue

            full_class_path = os.path.join(project_root, class_rel_path)
            if not os.path.exists(full_class_path):
                print(f"⚠️ No se encontró el archivo: {full_class_path}")
                continue

            print(f"\n📁 Refactorizando clase con Eclipse: {os.path.basename(full_class_path)}")

            # --- TRUCO CLAVE 1 MEJORADO: BOTTOM-UP & INNER-FIRST ---
            # Ordenamos de abajo hacia arriba (mayor start_offset primero).
            # Si dos rangos empiezan igual, el más corto (más anidado) va primero.
            raw_ranges.sort(key=lambda r: (-r[0], r[1] - r[0]))

            client.open_file(full_class_path)

            for i in range(len(raw_ranges)):
                current_range = raw_ranges[i]
                start_offset, end_offset = current_range

                # Leer el estado actual del archivo en disco
                with open(full_class_path, "r", encoding="utf-8", newline='') as f:
                    content = f.read()
                old_length = len(content)

                # Traducir offsets absolutos a posiciones LSP
                start_pos = offset_to_position(content, start_offset)
                end_pos = offset_to_position(content, end_offset)

                print(f"   ↳ Solicitando extracción a Eclipse en rango original: {start_offset}-{end_offset}")

                # Pedir a Eclipse que aplique la refactorización
                client.last_shifts = []  # Limpiar shifts anteriores
                success = client.request_extract_method(full_class_path, start_pos, end_pos)

                if success:
                    print(f"      ✅ Extracción exitosa. Registrados {len(client.last_shifts)} cambios locales.")

                    # Notificar a Eclipse que el archivo cambió para que re-compile internamente
                    client.close_file(full_class_path)
                    client.open_file(full_class_path)

                    # --- NUEVO TRUCO: AJUSTE DINÁMICO DE OFFSETS EXACTO ---
                    # Aplicamos los desplazamientos milimétricos a los offsets pendientes
                    for j in range(i + 1, len(raw_ranges)):
                        raw_ranges[j] = adjust_range_with_shifts(raw_ranges[j], client.last_shifts)
                else:
                    print("      ❌ Eclipse omitió esta extracción (no cumple condiciones semánticas).")

            client.close_file(full_class_path)

    finally:
        print("\n🧹 Deteniendo Eclipse JDT LS y limpiando workspace temporal...")
        client.stop()
        shutil.rmtree(temp_ws, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Refactorización AST por Lotes desde Pareto usando Eclipse JDT LS")
    parser.add_argument("--results-dir", required=True, help="Carpeta base de resultados (ej: MOILP_results_ASE)")
    parser.add_argument("--project-root", required=True, help="Ruta al proyecto Java (puede ser un .zip o una carpeta)")
    parser.add_argument("--algorithm", required=True, choices=["EpsilonConstraintAlgorithm", "HybridMethodAlgorithm"],
                        help="Algoritmo a procesar")
    parser.add_argument("--priority", nargs="+", default=["loc", "extractions", "cc"],
                        help="Orden de objetivos para el óptimo lexicográfico.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-class", help="Archivo de clase a refactorizar")
    group.add_argument("--all", action="store_true", help="Procesar todo el proyecto")

    args = parser.parse_args()

    # Gestión de archivos .zip
    project_root = args.project_root
    project_name = os.path.basename(os.path.normpath(project_root))

    if project_name.endswith(".zip"):
        clean_name = project_name[:-4]
        extraction_path = os.path.join(os.path.dirname(project_root), f"{clean_name}_unzipped")

        if os.path.exists(extraction_path):
            shutil.rmtree(extraction_path)

        with zipfile.ZipFile(project_root, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)

        inner_contents = os.listdir(extraction_path)
        if len(inner_contents) == 1 and os.path.isdir(os.path.join(extraction_path, inner_contents[0])):
            original_inner_folder = inner_contents[0]
            original_inner_path = os.path.join(extraction_path, original_inner_folder)
            new_inner_folder = f"{original_inner_folder}-refactored"
            new_inner_path = os.path.join(extraction_path, new_inner_folder)
            os.rename(original_inner_path, new_inner_path)
            project_root = new_inner_path
        else:
            project_root = extraction_path

        project_name = clean_name

    target_results_dir = os.path.join(args.results_dir, project_name, f"{project_name}-3-objectives-results")

    if not os.path.exists(target_results_dir):
        print(f"\n⚠️ Error: No se encontró la carpeta de 3 objetivos en:\n{target_results_dir}")
        return

    class_map = process_results(target_results_dir, args.algorithm, args.priority, args.target_class)

    if not class_map:
        print("\n⚠️ No se encontraron resultados que aplicar.")
        return

    print("\n--- INICIANDO REFACTORIZACIÓN EN LOTE CON ECLIPSE JDT LS ---")
    apply_refactorings_to_classes(class_map, project_root)


if __name__ == "__main__":
    main()