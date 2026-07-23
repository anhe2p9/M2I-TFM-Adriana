import os
import re
import ast
import urllib.parse
import sys
import json
import time
import argparse
import subprocess
import tempfile
import threading
import queue
import pandas as pd
from pathlib import Path
import zipfile
import shutil

# --- CONFIGURACIÓN DE ECLIPSE JDT LS ---
JDTLS_HOME = os.getenv("JDTLS_HOME")

if not JDTLS_HOME:
    print("❌ Error: La variable de entorno 'JDTLS_HOME' no está configurada.")
    sys.exit(1)


# =====================================================================
# CLIENTE LSP CON MULTITHREADING Y AUTORECUPERACIÓN (SELF-HEALING)
# =====================================================================
class JDTLSClient:
    def __init__(self, jdtls_home, workspace_dir, project_path, use_system_m2=False):
        self.jdtls_home = os.path.abspath(jdtls_home).replace("\\", "/")
        self.workspace_dir = os.path.abspath(workspace_dir).replace("\\", "/")
        self.project_path = os.path.abspath(project_path).replace("\\", "/")
        self.use_system_m2 = use_system_m2
        self.proc = None
        self.request_id = 1
        self.file_versions = {}
        self.msg_queue = queue.Queue()
        self.reader_thread = None
        self.running = False
        self.file_diagnostics = {}

    def start(self):
        import uuid
        import glob
        self.running = True
        self.msg_queue = queue.Queue()

        uid = uuid.uuid4().hex[:8]

        # 1. Clonar el servidor
        local_jdtls = os.path.join(tempfile.gettempdir(), f"jdtls_iso_{uid}")
        shutil.copytree(self.jdtls_home, local_jdtls)

        # 2. Permisos
        for root_dir, dirs, files in os.walk(local_jdtls):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root_dir, d), 0o777)
                except Exception:
                    pass
            for f in files:
                try:
                    os.chmod(os.path.join(root_dir, f), 0o777)
                except Exception:
                    pass

        # 3. Workspace
        jdtls_workspace = os.path.join(tempfile.gettempdir(), f"jdtls_ws_{uid}")
        os.makedirs(jdtls_workspace, exist_ok=True)
        try:
            os.chmod(jdtls_workspace, 0o777)
        except Exception:
            pass

        # 4. Encontrar Launcher
        plugins_dir = os.path.join(local_jdtls, "plugins")
        all_launchers = glob.glob(os.path.join(plugins_dir, "org.eclipse.equinox.launcher_*.jar"))
        valid_launchers = [f for f in all_launchers if not any(
            x in os.path.basename(f) for x in ["source", "gtk", "win32", "cocoa", "x86_64", "arm64"])]
        if not valid_launchers:
            valid_launchers = all_launchers
        if not valid_launchers:
            raise FileNotFoundError(f"❌ No se encontró launcher en {plugins_dir}")

        launcher_jar = os.path.abspath(valid_launchers[0]).replace("\\", "/")

        # 5. Validar Configuración
        config_dir = "config_win" if sys.platform.startswith("win") else "config_mac" if sys.platform.startswith(
            "darwin") else "config_linux"
        config_path = os.path.join(local_jdtls, config_dir).replace("\\", "/")

        # ¡NUEVO!: Si config_linux no tiene config.ini, usar la carpeta compartida o fallará con Código 13
        if not os.path.exists(os.path.join(config_path, "config.ini")):
            alt_config = os.path.join(local_jdtls, "config_ss_linux").replace("\\", "/")
            if os.path.exists(os.path.join(alt_config, "config.ini")):
                config_path = alt_config

        osgi_cache = os.path.join(config_path, "org.eclipse.osgi")
        if os.path.exists(osgi_cache):
            shutil.rmtree(osgi_cache, ignore_errors=True)

        cmd = [
            "C:/Program Files/Java/jdk-25/bin/java.exe",
            "-Djava.awt.headless=true",
            "-Declipse.application=org.eclipse.jdt.ls.core.id1",
            "-Dosgi.bundles.defaultStartLevel=4",
            "-Declipse.product=org.eclipse.jdt.ls.core.product",
            "-Dlog.level=ALL",
            "-Xmx2G",
            "--add-modules=ALL-SYSTEM",
            "--add-opens", "java.base/java.util=ALL-UNNAMED",
            "--add-opens", "java.base/java.lang=ALL-UNNAMED",
            "--add-opens", "java.base/sun.nio.ch=ALL-UNNAMED",
            "-jar", launcher_jar,
            "-configuration", config_path,
            "-data", jdtls_workspace,
            "-noconsole"
        ]

        # 2. Si NO se ha activado el flag, le añadimos la restricción /tmp
        if not self.use_system_m2:
            cmd.insert(1, "-Duser.home=/tmp")

        print(f"\n🚀 [Sistema] Iniciando JVM... (Buscando logs profundos OSGi en caso de error)")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.reader_thread = threading.Thread(target=self._enqueue_output, daemon=True)
        self.reader_thread.start()

        time.sleep(4.0)  # Damos margen para que Equinox vuelque el log

        if self.proc.poll() is not None:
            err_msg = self.proc.stderr.read().decode('utf-8', errors='ignore') if self.proc.stderr else ""
            out_msg = self.proc.stdout.read().decode('utf-8', errors='ignore') if self.proc.stdout else ""

            # Buscar logs tanto en configuración como en workspace
            log_contents = ""
            log_files = glob.glob(os.path.join(config_path, "*.log")) + glob.glob(
                os.path.join(jdtls_workspace, ".metadata", "*.log"))
            for lf in log_files:
                try:
                    with open(lf, 'r', encoding='utf-8', errors='ignore', newline="") as f:
                        content = f.read().strip()
                        if content:
                            log_contents += f"\n--- LOG ENCONTRADO: {os.path.basename(lf)} ---\n{content}\n"
                except Exception:
                    pass

            detail = f"--- STDOUT ---\n{out_msg}\n--- STDERR ---\n{err_msg}"
            if log_contents:
                detail += log_contents
            else:
                detail += f"\n[!] Tampoco se encontró log OSGi en {config_path}."

            raise RuntimeError(
                f"El proceso JDT LS ha fallado al iniciar (Código {self.proc.returncode}).\nDetalle:\n{detail}")

    def _enqueue_output(self):
        stdout = self.proc.stdout
        try:
            while self.running and self.proc.poll() is None:
                content_length = None
                while True:
                    line_bytes = stdout.readline()
                    if not line_bytes: return
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if not line:
                        if content_length is not None: break
                        continue
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":")[1].strip())

                if content_length is not None:
                    body = stdout.read(content_length).decode('utf-8', errors='ignore')
                    try:
                        self.msg_queue.put(json.loads(body))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

    def send(self, method, params=None, is_notification=False):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None: payload["params"] = params
        if not is_notification:
            payload["id"] = self.request_id
            self.request_id += 1

        body = json.dumps(payload)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        self.proc.stdin.write((header + body).encode('utf-8'))
        self.proc.stdin.flush()
        return payload.get("id")

    def read_message(self, timeout=0.1):
        try:
            msg = self.msg_queue.get(timeout=timeout)

            if msg.get("method") in ["window/logMessage", "language/status"]:
                text = msg.get("params", {}).get("message", "")
                if any(x in text for x in ["Compile", "Build", "Error", "Indexing"]):
                    print(f"      🛠️ [Eclipse] {text}")



            elif msg.get("method") == "textDocument/publishDiagnostics":
                uri = msg.get("params", {}).get("uri", "")
                diagnostics = msg.get("params", {}).get("diagnostics", [])
                errors = [d for d in diagnostics if d.get("severity") == 1]

                # --- DECODIFICACIÓN SEGURA DE URI PARA WINDOWS ---
                clean_uri = urllib.parse.unquote(uri)

                if clean_uri.startswith("file:///"):
                    clean_uri = clean_uri[8:]
                elif clean_uri.startswith("file:/"):
                    clean_uri = clean_uri[6:]

                norm_uri = os.path.abspath(clean_uri).lower()
                self.file_diagnostics[norm_uri] = errors
                # -------------------------------------------------

                if errors:
                    filename = os.path.basename(uri)
                    err_msg = errors[0]['message']
                    print(f"      🚨 [Error de Compilación en {filename}]: {err_msg}")

                    # 🧹 DETECCIÓN Y AUTO-LIMPIEZA DE JARS CORRUPTOS EN /tmp
                    if "not a valid ZIP file" in err_msg or "cannot be read" in err_msg:
                        match = re.search(r"'([^']+\.jar)'", err_msg)
                        if match:
                            corrupt_jar = match.group(1)
                            if os.path.exists(corrupt_jar):
                                try:
                                    os.remove(corrupt_jar)
                                    print(f"      🧹 [Auto-Fix] Se ha eliminado el JAR corrupto: {os.path.basename(corrupt_jar)}")
                                except Exception as e:
                                    print(f"      ⚠️ No se pudo eliminar el JAR corrupto: {e}")

            return msg
        except queue.Empty:
            return None

    def clear_file_errors(self, file_path):
        """Limpia el registro de errores previo para forzar una lectura fresca."""
        norm_path = os.path.abspath(file_path).lower()
        self.file_diagnostics[norm_path] = None

    def get_file_errors(self, file_path, timeout=5.0):
        """Espera a que Eclipse emita el diagnóstico DEFINITIVO del archivo."""
        norm_path = os.path.abspath(file_path).lower()
        start = time.time()

        # Drenamos la cola durante el tiempo COMPLETO.
        # Ignoramos el primer mensaje vacío [] de Eclipse y le damos tiempo real de compilar.
        while time.time() - start < timeout:
            self.read_message(timeout=0.2)

        res = self.file_diagnostics.get(norm_path)
        return res if res is not None else []

    def wait_for_response(self, req_id, desired_name=None, timeout=60):
        start_time, last_ping = time.time(), time.time()
        while time.time() - start_time < timeout:
            if time.time() - last_ping > 10:
                print(
                    f"      ⏳ [Esperando...] Eclipse sigue procesando ({(time.time() - start_time):.0f}s / {timeout}s)")
                last_ping = time.time()

            msg = self.read_message(timeout=0.5)
            if not msg: continue

            if msg.get("method") == "workspace/applyEdit":
                self.apply_workspace_edit(msg["params"]["edit"], desired_name)
                self.send_response(msg["id"], {"applied": True})
            elif msg.get("id") == req_id:
                return msg
        raise TimeoutError("⏳ El servidor Eclipse JDT LS no respondió a tiempo.")

    def send_response(self, req_id, result):
        payload = {"jsonrpc": "2.0", "id": req_id, "result": result}
        body = json.dumps(payload)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        try:
            self.proc.stdin.write((header + body).encode('utf-8'))
            self.proc.stdin.flush()
        except OSError:
            pass

    def initialize(self):
        print("\n☕ [Fase 1] Inicializando Eclipse JDT LS (Compilando e indexando proyecto)...")
        req_id = self.send("initialize", {
            "processId": os.getpid(),
            "rootUri": f"file:///{self.project_path}",
            "capabilities": {
                "workspace": {"applyEdit": True},
                "textDocument": {
                    "codeAction": {"codeActionLiteralSupport": {"codeActionKind": {"valueSet": ["refactor.extract"]}}}}
            }
        })
        self.wait_for_response(req_id, timeout=1000)
        self.send("initialized", is_notification=True)
        print("☕ [Fase 1 completada] Eclipse listo y proyecto cargado.")

    def open_file(self, file_path):
        abs_path = os.path.abspath(file_path).replace("\\", "/")

        try:
            with open(file_path, 'r', encoding='utf-8', newline="") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1', newline="") as f:
                content = f.read()

        self.file_versions[abs_path] = self.file_versions.get(abs_path, 0) + 1
        self.send("textDocument/didOpen", {
            "textDocument": {"uri": f"file:///{abs_path}", "languageId": "java",
                             "version": self.file_versions[abs_path], "text": content}
        }, is_notification=True)

    def close_file(self, file_path):
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        self.send("textDocument/didClose", {"textDocument": {"uri": f"file:///{abs_path}"}}, is_notification=True)

    def request_extract_method(self, file_path, start_pos, end_pos, desired_name, timeout=30):
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        req_id = self.send("textDocument/codeAction", {
            "textDocument": {"uri": f"file:///{abs_path}"},
            "range": {"start": start_pos, "end": end_pos},
            "context": {"diagnostics": [], "only": ["refactor.extract"]}
        })
        resp = self.wait_for_response(req_id, desired_name=desired_name, timeout=timeout)

        actions = resp.get("result", [])
        extract_action = next(
            (a for a in actions if "extract" in a.get("title", "").lower() and "method" in a.get("title", "").lower()),
            None)

        if not extract_action: return False

        if "command" in extract_action:
            cmd = extract_action["command"]
            r_id = self.send("workspace/executeCommand",
                             {"command": cmd["command"], "arguments": cmd.get("arguments", [])})
            self.wait_for_response(r_id, desired_name=desired_name, timeout=timeout)
            return True
        elif "edit" in extract_action:
            return self.apply_workspace_edit(extract_action["edit"], desired_name)
        return False

    def apply_workspace_edit(self, edit, desired_name):
        applied_all = True
        if "changes" in edit:
            for uri, text_edits in edit["changes"].items():
                if not self.apply_text_edits(uri.replace("file:///", "").replace("/", os.sep), text_edits,
                                             desired_name):
                    applied_all = False
        elif "documentChanges" in edit:
            for doc_change in edit["documentChanges"]:
                if "textDocument" in doc_change:
                    if not self.apply_text_edits(
                            doc_change["textDocument"]["uri"].replace("file:///", "").replace("/", os.sep),
                            doc_change["edits"], desired_name):
                        applied_all = False
        return applied_all

    def apply_text_edits(self, file_path, edits, desired_name):
        try:
            with open(file_path, 'r', encoding='utf-8', newline="") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1', newline="") as f:
                content = f.read()
        sorted_edits = sorted(edits, key=lambda e: (e["range"]["start"]["line"], e["range"]["start"]["character"]),
                              reverse=True)
        lines = content.splitlines(keepends=True)

        for edit in sorted_edits:
            start, end = edit["range"]["start"], edit["range"]["end"]
            new_text = re.sub(r"\bextracted\d*\b", desired_name, edit["newText"]) if desired_name else edit["newText"]

            # --- ESCUDO DE SEGURIDAD: Detectar el bug de Múltiples Variables de Salida de Eclipse ---
            # Busca declaraciones huérfanas (ej: "Object type;" o "int i;") justo antes del return o la llave de cierre.
            patron_huerfana = re.search(
                r"\b([A-Z][a-zA-Z0-9_<>,\[\]]*|int|boolean|double|float|long|short|byte|char)\s+([a-zA-Z0-9_$]+)\s*;\s*(?:return\s+[^;]+;)?\s*\}",
                new_text)

            if patron_huerfana:
                tipo_var = patron_huerfana.group(1)
                nombre_var = patron_huerfana.group(2)

                reporte = f"⚠️ [EXTRACCIÓN ABORTADA] Bug de JDT detectado (Múltiples Variables de Salida). " \
                          f"Eclipse dejó la variable '{tipo_var} {nombre_var}' fuera de ámbito. " \
                          f"Se omite la refactorización para evitar romper el código."

                print(reporte)

                # Si estás guardando reportes en un archivo, añádelo aquí. Ejemplo:
                # with open("extraction_reports.log", "a", encoding="utf-8") as log_file:
                #     log_file.write(f"{desired_name}: {reporte}\n")

                # Abortamos la aplicación de esta edición y notificamos el fallo
                return False
                # ----------------------------------------------------------------------------------------

            # --- LÓGICA MULTI-GENÉRICA (CORREGIDA PARA MODIFICADORES Y DUPLICADOS) ---
            if desired_name:
                # 1. Usamos [^{};=]+? para atrapar el tipo de retorno de forma ultra-segura.
                #    Esto evita que la expresión regular retroceda y mutile declaraciones de
                #    clases (public class...), enums o campos globales.
                patron_firma = re.compile(
                    r"((?:(?:public|protected|private|static|final|synchronized|native|strictfp)\s+)+)([^{};=]+?)(\s+)([a-zA-Z_$][a-zA-Z0-9_$]*)(\s*\()([^)]*)(\))"
                )

                def inyectar_genericos(match):
                    modificadores = match.group(1)
                    tipo_retorno_raw = match.group(2)
                    espacio = match.group(3)
                    nombre_metodo = match.group(4)
                    apertura = match.group(5)
                    parametros = match.group(6)
                    cierre = match.group(7)

                    tipo_retorno_limpio = tipo_retorno_raw.strip()

                    if tipo_retorno_limpio.startswith("<"):
                        return match.group(0)

                    texto_analisis = f"{tipo_retorno_limpio} {parametros}"
                    letras_genericas = re.findall(r"\b[A-Z]\b", texto_analisis)
                    letras_unicas = list(dict.fromkeys(letras_genericas))

                    if letras_unicas:
                        declaracion = f"<{', '.join(letras_unicas)}> "
                        return f"{modificadores}{declaracion}{tipo_retorno_limpio}{espacio}{nombre_metodo}{apertura}{parametros}{cierre}"

                    return match.group(0)

                # Aplicamos la sustitución usando el patrón blindado
                new_text = patron_firma.sub(inyectar_genericos, new_text)
            # ------------------------------------------------

            s_line, s_char, e_line, e_char = start["line"], start["character"], end["line"], end["character"]

            if s_line == e_line:
                lines[s_line] = lines[s_line][:s_char] + new_text + lines[s_line][e_char:]
            else:
                lines[s_line] = lines[s_line][:s_char] + new_text + lines[e_line][e_char:]
                for _ in range(s_line + 1, e_line + 1): lines.pop(s_line + 1)

        try:
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write("".join(lines))
        except UnicodeEncodeError:
            with open(file_path, "w", encoding="latin-1", newline="") as f:
                f.write("".join(lines))

        return True

    def stop(self):
        self.running = False
        if self.proc and self.proc.poll() is None:
            try:
                self.send("shutdown")
                self.send("exit", is_notification=True)
            except:
                pass
            finally:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()


# =====================================================================
# LÓGICA DE PARSEO Y MANEJO ESTRATÉGICO DE MARCADORES (ÁRBOLES)
# =====================================================================
def offset_to_position(content, offset):
    before = content[:offset]
    lines = before.split('\n')
    return {"line": len(lines) - 1, "character": len(lines[-1])}


def process_results(results_base_dir, target_algo, user_priority, target_class=None):
    class_offsets_map = {}
    base_path = Path(results_base_dir)

    # Expresión regular adaptada para soportar cualquier número de objetivos (2 ó 3)
    folder_pattern = re.compile(r"^(?P<algo>[^_]+)_(?P<objs>[^_]+)_(?P<classpath>.*\.java)_(?P<method>.+)$")

    for root, dirs, files in os.walk(base_path):
        current_folder = os.path.basename(root)
        match = folder_pattern.match(current_folder)

        if match:
            algo, objs_str, classpath, method = match.group("algo"), match.group("objs"), match.group(
                "classpath"), match.group("method")

            if algo != target_algo: continue

            # --- FILTRADO ESTRICTO PARA 3 OBJETIVOS ---
            objs_list = [o.lower() for o in objs_str.split('-')]

            # 1. Ignorar si no tiene exactamente 3 objetivos (descarta las ejecuciones de 2 objetivos)
            if len(objs_list) != 3:
                continue

            # 2. Validar que los objetivos sean componentes válidos
            valid_objs = {'ex', 'extractions', 'cc', 'loc'}
            if not all(o in valid_objs for o in objs_list):
                continue
            # ------------------------------------------
            if target_class and not classpath.endswith(target_class): continue

            csv_path = os.path.join(root, f"{method}_complete_data.csv")
            if not os.path.exists(csv_path): continue

            try:
                df = pd.read_csv(csv_path)
                df['parsed_solution'] = df['solution'].apply(ast.literal_eval)
            except Exception:
                continue

            df = df[df['parsed_solution'].apply(lambda x: len(x) > 0)]
            if df.empty: continue

            folder_objs = ['extractions' if o == 'ex' else o for o in [obj.lower() for obj in objs_str.split('-')]]
            usr_prio = ['extractions' if p == 'ex' else p for p in [p.lower() for p in user_priority]]
            sort_indices = [folder_objs.index(p) for p in usr_prio if p in folder_objs] or list(range(len(folder_objs)))

            df_sorted = df.sort_values(by='parsed_solution', key=lambda col: col.map(
                lambda x: tuple(x[i] for i in sort_indices if i < len(x))))
            best_row = df_sorted.iloc[0]

            extracted_offsets = ast.literal_eval(str(best_row['offsets'])) if 'offsets' in best_row else []
            info_str = str(best_row.get('solution_info (index,CC,LOC)', '[]'))
            nested_str = str(best_row.get('nested_solution', '{}'))

            try:
                info_list = ast.literal_eval(info_str)
            except:
                info_list = []

            try:
                nested_dict = ast.literal_eval(nested_str)
            except:
                nested_dict = {}

            ext_indices = [tup[0] for tup in info_list[1:]] if len(info_list) > 1 else []
            method_extractions = []
            for i, rng in enumerate(extracted_offsets):
                if not isinstance(rng, list) or len(rng) != 2: continue
                orig_idx = ext_indices[i] if i < len(ext_indices) else None
                method_extractions.append({"range": rng, "orig_idx": orig_idx})

            def get_depth(idx, current_depth=0):
                for parent, children in nested_dict.items():
                    if idx in children: return get_depth(parent, current_depth + 1)
                return current_depth

            for ext in method_extractions:
                ext["depth"] = get_depth(ext["orig_idx"]) if ext["orig_idx"] is not None else 0

            # Normalizar la ruta de la clase a partir del classpath
            if classpath.endswith(".java"):
                base_cp = classpath[:-5]
            else:
                base_cp = classpath
            real_class_path = base_cp.replace("-", "/").replace(".", "/") + ".java"

            # Limpiamos el guión y el número de línea del nombre del método
            clean_method = method.split('-')[0]

            if real_class_path not in class_offsets_map: class_offsets_map[real_class_path] = {}
            if clean_method not in class_offsets_map[real_class_path]: class_offsets_map[real_class_path][
                clean_method] = []
            class_offsets_map[real_class_path][clean_method].extend(method_extractions)

    return class_offsets_map


def prepare_extractions_with_names(methods_dict, class_content):
    prepared = []
    for method_name, ext_list in methods_dict.items():
        sorted_by_size = sorted(ext_list, key=lambda e: e["range"][1] - e["range"][0])
        extraction_n = 1

        for ext in sorted_by_size:
            desired_name = f"{method_name}_extraction_{extraction_n}"

            # Comprobar si el nombre ya existe en el código fuente de la clase
            while f"{desired_name}(" in class_content or f"{desired_name} (" in class_content:
                extraction_n += 1
                desired_name = f"{method_name}_extraction_{extraction_n}"

            prepared.append(
                {"range": ext["range"], "depth": ext["depth"], "desired_name": desired_name,
                 "method_name": method_name})

            # Incrementar para la siguiente extracción del mismo método
            extraction_n += 1

    prepared.sort(key=lambda x: x["range"][1] - x["range"][0])
    for ext_id, item in enumerate(prepared): item["ext_id"] = ext_id
    return prepared


def inject_markers(file_path, prepared_extractions):
    insertions = []
    for item in prepared_extractions:
        r, ext_id, depth = item["range"], item["ext_id"], item["depth"]
        insertions.append({"pos": r[1], "text": f"/*END_EXT_{ext_id}*/", "is_start": False, "depth": depth})
        insertions.append({"pos": r[0], "text": f"/*START_EXT_{ext_id}*/", "is_start": True, "depth": depth})

    def sort_key(ins):
        p1 = -ins["pos"]
        p2 = 1 if ins["is_start"] else 0
        p3 = -ins["depth"] if ins["is_start"] else ins["depth"]
        return (p1, p2, p3)

    insertions.sort(key=sort_key)

    encoding_usado = 'utf-8'
    try:
        with open(file_path, 'r', encoding='utf-8', newline="") as f:
            content = f.read()
    except UnicodeDecodeError:
        encoding_usado = 'latin-1'
        with open(file_path, 'r', encoding='latin-1', newline="") as f:
            content = f.read()
    for ins in insertions: content = content[:ins["pos"]] + ins["text"] + content[ins["pos"]:]
    with open(file_path, "w", encoding=encoding_usado, newline="") as f:
        f.write(content)


def sanitize_uninitialized_variables(content):
    """
    Parche definitivo anti-bug de Eclipse JDT LS:
    1. Desglosa declaraciones múltiples en LÍNEAS INDEPENDIENTES.
       Ej: 'int i, j;' -> 'int i = 0;\nint j = 0;'
       Esto evita que Eclipse borre la línea entera si decide mover una de las variables.
    2. Inicializa todas las variables locales con valores por defecto.
    """
    lines = content.splitlines()
    new_lines = []

    # Patrón para preservar indentación y marcadores comentarios /*START...*/
    prefijo_regex = r'^(\s*(?:/\*.*?\*/\s*)*)'

    for line in lines:
        # Ignoramos si ya está inicializada, es 'final', 'return', o import/package
        if 'final ' in line or 'return ' in line or '=' in line or line.strip().startswith(('import ', 'package ')):
            new_lines.append(line)
            continue

        # 1. Arrays (ej: int[] a, b; o String[] x, y;)
        m = re.match(prefijo_regex + r'([a-zA-Z0-9_<>\?]+\s*\[\])\s+([a-zA-Z0-9_$,\s]+);(.*)$', line)
        if m:
            pref, tipo, vars_str, rest = m.groups()
            vars_list = [v.strip() for v in vars_str.split(',') if v.strip()]
            for idx, v in enumerate(vars_list):
                comment = rest if idx == len(vars_list) - 1 else ""
                new_lines.append(f"{pref}{tipo} {v} = null;{comment}")
            continue

        # 2. Primitivos numéricos (ej: int i, j; float x, y; int b1, b2, b3;)
        m = re.match(prefijo_regex + r'(int|double|float|long|short|byte|char)\s+([a-zA-Z0-9_$,\s]+);(.*)$', line)
        if m:
            pref, tipo, vars_str, rest = m.groups()
            vars_list = [v.strip() for v in vars_str.split(',') if v.strip()]
            for idx, v in enumerate(vars_list):
                comment = rest if idx == len(vars_list) - 1 else ""
                new_lines.append(f"{pref}{tipo} {v} = 0;{comment}")
            continue

        # 3. Booleanos (ej: boolean flag, done;)
        m = re.match(prefijo_regex + r'(boolean)\s+([a-zA-Z0-9_$,\s]+);(.*)$', line)
        if m:
            pref, tipo, vars_str, rest = m.groups()
            vars_list = [v.strip() for v in vars_str.split(',') if v.strip()]
            for idx, v in enumerate(vars_list):
                comment = rest if idx == len(vars_list) - 1 else ""
                new_lines.append(f"{pref}{tipo} {v} = false;{comment}")
            continue

        # 4. Objetos y Genéricos (ej: Collection a, b; o String s1, s2;)
        m = re.match(prefijo_regex + r'([A-Z][a-zA-Z0-9_<>,\?\[\]]*)\s+([a-zA-Z0-9_$,\s]+);(.*)$', line)
        if m:
            pref, tipo, vars_str, rest = m.groups()
            if '(' not in vars_str and ')' not in vars_str:
                vars_list = [v.strip() for v in vars_str.split(',') if v.strip()]
                for idx, v in enumerate(vars_list):
                    comment = rest if idx == len(vars_list) - 1 else ""
                    new_lines.append(f"{pref}{tipo} {v} = null;{comment}")
                continue

        new_lines.append(line)

    return "\n".join(new_lines)


# =====================================================================
# MOTOR PRINCIPAL DE REFACTORIZACIÓN CON SEGUIMIENTO DE MÉTRICAS
# =====================================================================
def apply_refactorings_to_classes(class_offsets_map, project_root, use_system_m2=False):
    temp_ws = tempfile.mkdtemp(prefix="jdtls_ws_")
    client = JDTLSClient(JDTLS_HOME, temp_ws, project_root, use_system_m2=use_system_m2)
    extraction_results = []

    try:
        client.start()
        client.initialize()

        for class_rel_path, methods_dict in class_offsets_map.items():
            if not methods_dict: continue

            # 1. Intentar la ruta relativa completa
            full_class_path = os.path.join(project_root, class_rel_path)

            # 2. Si no existe, buscar por coincidencia de subcarpeta o nombre de archivo (.java)
            if not os.path.exists(full_class_path):
                found_path = None
                target_filename = os.path.basename(class_rel_path)

                for current_dir, dirs, files in os.walk(project_root):
                    if target_filename in files:
                        found_path = os.path.join(current_dir, target_filename)
                        break

                if found_path:
                    full_class_path = found_path
                else:
                    print(
                        f"   👻 Archivo fantasma: No se encontró '{class_rel_path}' ({target_filename}) en {os.path.basename(project_root)}")
                    continue

            print(f"\n📁 Procesando clase: {os.path.basename(full_class_path)}")

            # --- 1. Leer el archivo intacto para no romper los offsets originales ---
            try:
                with open(full_class_path, 'r', encoding='utf-8', newline="") as f:
                    class_content = f.read()
            except UnicodeDecodeError:
                with open(full_class_path, 'r', encoding='latin-1', newline="") as f:
                    class_content = f.read()

            # --- 2. Inyectar marcadores USANDO LOS OFFSETS INTACTOS ---
            prepared_extractions = prepare_extractions_with_names(methods_dict, class_content)
            inject_markers(full_class_path, prepared_extractions)
            print(f"   ↳ Inyectados marcadores jerárquicos para {len(prepared_extractions)} bloques.")

            # --- 3. Leer el archivo YA MARCADO ---
            encoding_usado = 'utf-8'
            try:
                with open(full_class_path, 'r', encoding='utf-8', newline="") as f:
                    marked_content = f.read()
            except UnicodeDecodeError:
                encoding_usado = 'latin-1'
                with open(full_class_path, 'r', encoding='latin-1', newline="") as f:
                    marked_content = f.read()

            # --- 4. Aplicar el saneamiento de variables sobre el código ya marcado ---
            # Al hacerlo ahora, el cambio de longitud del archivo no desalinea ningún offset
            sanitized_content = sanitize_uninitialized_variables(marked_content)

            with open(full_class_path, "w", encoding=encoding_usado, newline="") as f:
                f.write(sanitized_content)

            time.sleep(3)
            client.open_file(full_class_path)

            for ext_item in prepared_extractions:
                ext_id, desired_name = ext_item["ext_id"], ext_item["desired_name"]

                encoding_usado = 'utf-8'
                try:
                    with open(full_class_path, 'r', encoding='utf-8', newline="") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    encoding_usado = 'latin-1'
                    with open(full_class_path, 'r', encoding='latin-1', newline="") as f:
                        content = f.read()

                # --- RESPALDO PREVIO PARA ROLLBACK ---
                backup_content_clean = content

                start_marker, end_marker = f"/*START_EXT_{ext_id}*/", f"/*END_EXT_{ext_id}*/"
                start_idx, end_idx = content.find(start_marker), content.find(end_marker)

                if start_idx == -1 or end_idx == -1: continue

                raw_code_snippet = content[start_idx + len(start_marker): end_idx].strip()
                snippet_lines = raw_code_snippet.splitlines()
                preview_text = snippet_lines[0] if snippet_lines else ""
                if len(preview_text) > 80: preview_text = preview_text[:77] + "..."

                content = content[:end_idx] + content[end_idx + len(end_marker):]
                content = content[:start_idx] + content[start_idx + len(start_marker):]

                with open(full_class_path, "w", encoding=encoding_usado, newline="") as f:
                    f.write(content)

                # Limpiamos diagnósticos previos antes de pedir la extracción
                client.clear_file_errors(full_class_path)
                start_pos, end_pos = offset_to_position(content, start_idx), offset_to_position(content,
                                                                                                end_idx - len(
                                                                                                    start_marker))

                print(f"\n👉 [Extracción {ext_id + 1}/{len(prepared_extractions)}] Asignando: '{desired_name}'")
                print(f"   📝 Código: \"{preview_text}\"")

                client.send("textDocument/didClose",
                            {"textDocument": {"uri": f"file:///{full_class_path.replace('\\', '/')}"}},
                            is_notification=True)
                client.send("textDocument/didOpen", {
                    "textDocument": {"uri": f"file:///{full_class_path.replace('\\', '/')}", "languageId": "java",
                                     "version": int(time.time()), "text": content}
                }, is_notification=True)

                success = False
                max_intentos = 2  # 1 intento original + 1 reintento tras recuperación

                for intento in range(1, max_intentos + 1):
                    try:
                        success = client.request_extract_method(full_class_path, start_pos, end_pos, desired_name,
                                                                timeout=100)
                        break
                    except TimeoutError:
                        print(f"\n⚠️ [Timeout] Eclipse JDT LS se ha colgado (Intento {intento}/{max_intentos}).")

                        if intento < max_intentos:
                            print("   Iniciando proceso de autorecuperación para reintentar...")
                            client.stop()
                            time.sleep(2)
                            shutil.rmtree(client.workspace_dir, ignore_errors=True)
                            os.makedirs(client.workspace_dir, exist_ok=True)

                            client.start()
                            client.initialize()
                            client.open_file(full_class_path)
                            print("      🔄 Servidor recuperado con éxito. Ejecutando reintento...\n")
                        else:
                            print(
                                "      ❌ Se han agotado los reintentos. Omitiendo esta extracción.")
                            success = False

                if success:
                    # --- EL CAMBIO CLAVE: Forzar a Eclipse a evaluar el NUEVO código ---
                    # Leemos el código tal y como ha quedado tras la extracción
                    with open(full_class_path, "r", encoding=encoding_usado, newline="") as f:
                        new_content = f.read()

                    # Vaciamos la memoria antigua de Eclipse y le inyectamos la nueva versión
                    client.send("textDocument/didClose",
                                {"textDocument": {"uri": f"file:///{full_class_path.replace('\\', '/')}"}},
                                is_notification=True)
                    client.send("textDocument/didOpen", {
                        "textDocument": {"uri": f"file:///{full_class_path.replace('\\', '/')}", "languageId": "java",
                                         "version": int(time.time()), "text": new_content}
                    }, is_notification=True)
                    # ------------------------------------------------------------------

                    # Ahora sí, Eclipse analiza el código modificado y esperamos su veredicto (3.5s)
                    errors = client.get_file_errors(full_class_path, timeout=5.0)

                    if not errors:
                        print("      ✅ ¡Extracción exitosa!")
                        extraction_results.append(
                            {"Clase": os.path.basename(full_class_path), "Metodo Original": ext_item["method_name"],
                             "Nombre Extraccion": desired_name, "Exito": "Sí"})
                    else:
                        print(f"      🚨 La extracción generó {len(errors)} error(es) de compilación en Eclipse.")
                        print(f"         ↳ Detalle: {errors[0].get('message')}")
                        success = False  # Rechazamos la extracción para forzar el Rollback

                if not success:
                    print("      ❌ Deshaciendo extracción (Rollback) para mantener la clase funcional...")
                    # 1. Restauramos el código previo
                    with open(full_class_path, "w", encoding=encoding_usado, newline="") as f:
                        f.write(backup_content_clean)

                    # 2. Refrescamos el estado en Eclipse enviando la versión limpia
                    client.send("textDocument/didClose",
                                {"textDocument": {"uri": f"file:///{full_class_path.replace('\\', '/')}"}},
                                is_notification=True)
                    client.send("textDocument/didOpen", {
                        "textDocument": {"uri": f"file:///{full_class_path.replace('\\', '/')}", "languageId": "java",
                                         "version": int(time.time()), "text": backup_content_clean}
                    }, is_notification=True)

                    extraction_results.append(
                        {"Clase": os.path.basename(full_class_path), "Metodo Original": ext_item["method_name"],
                         "Nombre Extraccion": desired_name, "Exito": "No"})

            client.close_file(full_class_path)

        return extraction_results

    finally:
        client.stop()
        shutil.rmtree(temp_ws, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Refactorización AST Jerárquica con Eclipse JDT LS")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--algorithm", required=True, choices=["EpsilonConstraintAlgorithm", "HybridMethodAlgorithm"])
    parser.add_argument("--priority", nargs="+", default=["loc", "extractions", "cc"])
    parser.add_argument("--use-system-m2", action="store_true", help="Usa la carpeta .m2 del usuario en vez de /tmp")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-class")
    group.add_argument("--all", action="store_true")

    args = parser.parse_args()

    project_root, project_name = args.project_root, os.path.basename(os.path.normpath(args.project_root))

    if project_name.endswith(".zip"):
        clean_name = project_name[:-4]
        extraction_path = os.path.join(os.path.dirname(project_root), f"{clean_name}_refactored_{args.algorithm}")
        if os.path.exists(extraction_path): shutil.rmtree(extraction_path)
        with zipfile.ZipFile(project_root, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)

        inner_contents = os.listdir(extraction_path)
        if len(inner_contents) == 1 and os.path.isdir(os.path.join(extraction_path, inner_contents[0])):
            os.rename(os.path.join(extraction_path, inner_contents[0]),
                      os.path.join(extraction_path, f"{inner_contents[0]}-refactored_{args.algorithm}"))
            project_root = os.path.join(extraction_path, f"{inner_contents[0]}-refactored_{args.algorithm}")
        else:
            project_root = extraction_path
        project_name = clean_name

    # --- BÚSQUEDA ADAPTATIVA INTELIGENTE Y CASE-INSENSITIVE ---
    target_project_name = project_name.lower()
    project_dir = None

    # 1. Buscar la carpeta del proyecto (soporta minusculas/mayusculas y rutas anidadas como output/results/fastjson)
    if os.path.basename(os.path.normpath(args.results_dir)).lower() == target_project_name:
        project_dir = args.results_dir
    else:
        for root, dirs, _ in os.walk(args.results_dir):
            for d in dirs:
                if d.lower() == target_project_name:
                    project_dir = os.path.join(root, d)
                    break
            if project_dir:
                break

    target_results_dir = None

    if project_dir:
        # 2. Si existe la subcarpeta explícita de "3-objectives", usamos esa
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path) and "3-objective" in item.lower():
                target_results_dir = item_path
                break

        # 3. Si no existe subcarpeta de "3-objectives" (caso 1), usamos la carpeta del proyecto directamente
        if not target_results_dir:
            target_results_dir = project_dir

    if not target_results_dir or not os.path.exists(target_results_dir):
        print(
            f"\n⚠️ Error: No se encontró ninguna carpeta correspondiente a '{project_name}' dentro de:\n{args.results_dir}")
        print("Comprueba que el nombre del proyecto coincida con la carpeta de resultados.")
        return

    print(f"🔍 Explorando resultados en: {target_results_dir}")
    class_map = process_results(target_results_dir, args.algorithm, args.priority, args.target_class)

    if not class_map:
        print("\n⚠️ No se encontraron resultados válidos para aplicar.")
        return

    print("\n--- INICIANDO REFACTORIZACIÓN EN LOTE ---")
    extraction_results = apply_refactorings_to_classes(class_map, project_root, use_system_m2=args.use_system_m2)

    total_intentos = len(extraction_results)

    if total_intentos > 0:
        exitosos = sum(1 for r in extraction_results if r["Exito"] == "Sí")
        fallidos = total_intentos - exitosos
        porcentaje = (exitosos / total_intentos) * 100

        parent_dir = os.path.dirname(os.path.normpath(args.project_root))
        csv_filename = f"{project_name}_metricas_extraccion.csv"
        csv_path = os.path.abspath(os.path.join(parent_dir, csv_filename))

        df_results = pd.DataFrame(extraction_results)
        df_results.to_csv(csv_path, index=False, encoding="utf-8")

        # --- AGRUPACIÓN A NIVEL DE MÉTODO ---
        method_stats = []
        for (clase, metodo), group in df_results.groupby(["Clase", "Metodo Original"]):
            total_ext = len(group)
            exitos = sum(group["Exito"] == "Sí")
            if exitos == total_ext:
                estado = "Completado"
            elif exitos == 0:
                estado = "Fallido"
            else:
                estado = f"Parcial ({exitos}/{total_ext})"

            method_stats.append({
                "Clase": clase,
                "Metodo Original": metodo,
                "Extracciones Previstas": total_ext,
                "Extracciones Exitosas": exitos,
                "Estado": estado
            })

        df_methods = pd.DataFrame(method_stats)
        csv_methods_path = os.path.abspath(os.path.join(parent_dir, f"{project_name}_metricas_metodos.csv"))
        df_methods.to_csv(csv_methods_path, index=False, encoding="utf-8")

        # Cálculos para el resumen en consola
        total_metodos = len(method_stats)
        metodos_completos = sum(1 for m in method_stats if m["Estado"] == "Completado")
        metodos_parciales = sum(1 for m in method_stats if m["Estado"].startswith("Parcial"))
        metodos_fallidos = sum(1 for m in method_stats if m["Estado"] == "Fallido")

        print("\n" + "=" * 70)
        print("🎉 REFACTORIZACIÓN COMPLETADA 🎉".center(70))
        print("=" * 70)

        print("\n📊 RENDIMIENTO POR EXTRACCIONES:")
        print(f"   * Total Intentadas: {total_intentos}")
        print(f"   * Exitosas: {exitosos}")
        print(f"   * Fallidas / Ignoradas: {fallidos}")
        print(f"   * Tasa de Éxito Global: **{porcentaje:.2f}%**")

        print("\n📊 RENDIMIENTO POR MÉTODOS ORIGINALES:")
        print(f"   * Total de Métodos Procesados: {total_metodos}")
        print(f"   * Refactorizados Completamente: {metodos_completos}")
        print(f"   * Refactorizados Parcialmente: {metodos_parciales}")
        print(f"   * Fallidos (Ninguna extracción): {metodos_fallidos}")

        print("\n📂 RUTAS DE SALIDA:")
        print(f"   * Proyecto Refactorizado: {os.path.abspath(project_root)}")
        print(f"   * Métricas de Extracciones (CSV): {csv_path}")
        print(f"   * Métricas de Métodos (CSV): {csv_methods_path}")
        print("\n" + "=" * 70 + "\n")
    else:
        print("\n⚠️ No se procesó ninguna extracción durante la ejecución.")


if __name__ == "__main__":
    main()