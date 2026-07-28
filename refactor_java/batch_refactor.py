import os
import re
import ast
import javalang
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


class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        # Abrimos el archivo en modo "append" (añadir) o "w" (sobrescribir)
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush() # Fuerza el guardado inmediato en disco

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


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
        self.last_edit_applied = True

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
            "-clearPersistedState",
            "-noconsole"
        ]

        # 2. Si NO se ha activado el flag, le añadimos la restricción /tmp
        if not self.use_system_m2:
            cmd.insert(1, "-Duser.home=/tmp")

        print(f"\n🚀 [Sistema] Iniciando JVM... (Buscando logs profundos OSGi en caso de error)")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.__stderr__)

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
                    try:
                        with open(lf, 'r', encoding='utf-8', newline="") as f:
                            content = f.read().strip()
                            if content:
                                log_contents += f"\n--- LOG ENCONTRADO: {os.path.basename(lf)} ---\n{content}\n"
                    except UnicodeDecodeError:
                        with open(lf, 'r', encoding='latin-1', newline="") as f:
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
        if params is not None: 
            payload["params"] = params
        if not is_notification:
            payload["id"] = self.request_id
            self.request_id += 1

        # Convertir a bytes primero para calcular la longitud exacta en bytes
        body_bytes = json.dumps(payload).encode('utf-8')
        header_bytes = f"Content-Length: {len(body_bytes)}\r\n\r\n".encode('utf-8')
        
        self.proc.stdin.write(header_bytes + body_bytes)
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

                    # --- NUEVO: Extraer la línea exacta del error ---
                    line_num = errors[0].get('range', {}).get('start', {}).get('line', 0) + 1

                    print(f"      🚨 [Error de Compilación en {filename} (Línea {line_num})]: {err_msg}")

                    # 🧹 DETECCIÓN Y AUTO-LIMPIEZA DE JARS CORRUPTOS EN /tmp
                    if "not a valid ZIP file" in err_msg or "cannot be read" in err_msg:
                        match = re.search(r"'([^']+\.jar)'", err_msg)
                        if match:
                            corrupt_jar = match.group(1)
                            if os.path.exists(corrupt_jar):
                                try:
                                    os.remove(corrupt_jar)
                                    print(
                                        f"      🧹 [Auto-Fix] Se ha eliminado el JAR corrupto: {os.path.basename(corrupt_jar)}")
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
    
    def sync_file_and_get_errors(self, file_path, timeout=5.0):
        """Fuerza a Eclipse a descartar su caché y re-indexar el código real en disco."""
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        
        try:
            with open(file_path, 'r', encoding='utf-8', newline="") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1', newline="") as f:
                content = f.read()

        # Re-notificar a Eclipse
        self.send("textDocument/didClose", {"textDocument": {"uri": f"file:///{abs_path}"}}, is_notification=True)
        self.clear_file_errors(file_path)
        
        self.send("textDocument/didOpen", {
            "textDocument": {
                "uri": f"file:///{abs_path}", 
                "languageId": "java",
                "version": int(time.time() * 1000), 
                "text": content
            }
        }, is_notification=True)

        return self.get_file_errors(file_path, timeout=timeout)

    def wait_for_response(self, req_id, desired_name=None, timeout=60, req_start_pos=None, req_end_pos=None):
        start_time, last_ping = time.time(), time.time()
        while time.time() - start_time < timeout:
            if time.time() - last_ping > 10:
                print(
                    f"      ⏳ [Esperando...] Eclipse sigue procesando ({(time.time() - start_time):.0f}s / {timeout}s)")
                last_ping = time.time()

            msg = self.read_message(timeout=0.5)
            if not msg: continue

            if msg.get("method") == "workspace/applyEdit":
                # Se pasa req_start_pos y req_end_pos de forma explícita
                exito_edicion = self.apply_workspace_edit(msg["params"]["edit"],
                                                           desired_name,
                                                            req_start_pos=req_start_pos,
                                                            req_end_pos=req_end_pos)
                self.last_edit_applied = exito_edicion # Guardamos el estado real
                self.send_response(msg["id"], {"applied": True})
            elif msg.get("id") == req_id:
                return msg
        raise TimeoutError("⏳ El servidor Eclipse JDT LS no respondió a tiempo.")

    def send_response(self, req_id, result):
        payload = {"jsonrpc": "2.0", "id": req_id, "result": result}
        
        body_bytes = json.dumps(payload).encode('utf-8')
        header_bytes = f"Content-Length: {len(body_bytes)}\r\n\r\n".encode('utf-8')
        
        try:
            self.proc.stdin.write(header_bytes + body_bytes)
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

    def request_document_symbols(self, file_path, timeout=10.0):
        """Pide a Eclipse JDT LS el AST de la clase."""
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        req_id = self.send("textDocument/documentSymbol", {
            "textDocument": {"uri": f"file:///{abs_path}"}
        })
        try:
            resp = self.wait_for_response(req_id, timeout=timeout)
            return resp.get("result", [])
        except TimeoutError:
            return []

    def find_method_line_in_symbols(self, symbols, target_method_name):
        """Navega por el AST recursivamente buscando la línea donde se declara el método."""
        for sym in symbols:
            # kind 6 = Method, kind 9 = Constructor
            if sym.get("kind") in (6, 9) and sym.get("name") == target_method_name:
                # LSP devuelve índices basados en 0
                return sym.get("range", {}).get("start", {}).get("line")
            
            # Buscar en clases anidadas
            if "children" in sym:
                res = self.find_method_line_in_symbols(sym["children"], target_method_name)
                if res is not None:
                    return res
        return None

    def close_file(self, file_path):
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        self.send("textDocument/didClose", {"textDocument": {"uri": f"file:///{abs_path}"}}, is_notification=True)

    def request_extract_method(self, file_path, start_pos, end_pos, desired_name, timeout=30, auto_extracted_ids=None):
        abs_path = os.path.abspath(file_path).replace("\\", "/")
        req_id = self.send("textDocument/codeAction", {
            "textDocument": {"uri": f"file:///{abs_path}"},
            "range": {"start": start_pos, "end": end_pos},
            "context": {"diagnostics": [], "only": ["refactor.extract"]}
        })
        resp = self.wait_for_response(req_id, desired_name=desired_name, timeout=timeout, req_start_pos=start_pos, req_end_pos=end_pos)

        actions = resp.get("result", [])
        extract_action = next(
            (a for a in actions if "extract" in a.get("title", "").lower() and "method" in a.get("title", "").lower()),
            None)

        if not extract_action: return False

        if "command" in extract_action:
            cmd = extract_action["command"]
            args = cmd.get("arguments", [])

            # --- DESACTIVAR REEMPLAZO DE DUPLICADOS EN ECLIPSE JDT LS ---
            options_dict = {"replaceAllOccurrences": False, "replaceDuplicates": False, "name": desired_name}

            # Buscar si Eclipse ya ha incluido un diccionario de opciones
            has_options = False
            for arg in args:
                # Evitar modificar el objeto 'range' que contiene 'start' y 'end'
                if isinstance(arg, dict) and "start" not in arg:
                    arg["replaceAllOccurrences"] = False
                    arg["replaceDuplicates"] = False
                    if desired_name:
                        arg["name"] = desired_name
                    has_options = True
                    break

            if not has_options:
                args.append(options_dict)
            # -------------------------------------------------------------------------

            r_id = self.send("workspace/executeCommand",
                             {"command": cmd["command"], "arguments": args})
            self.wait_for_response(r_id, desired_name=desired_name, timeout=timeout, req_start_pos=start_pos, req_end_pos=end_pos)
            if not self.last_edit_applied:
                return False
            return True
        elif "edit" in extract_action:
            # --- CORRECCIÓN 2: PASAR COORDENADAS COMO KEYWORD ARGUMENTS ---
            return self.apply_workspace_edit(
                extract_action["edit"], 
                desired_name, 
                auto_extracted_ids=auto_extracted_ids, 
                req_start_pos=start_pos, 
                req_end_pos=end_pos
            )
        return False

    def apply_workspace_edit(self, edit, desired_name, auto_extracted_ids=None, req_start_pos=None, req_end_pos=None):
        applied_all = True
        if "changes" in edit:
            for uri, text_edits in edit["changes"].items():
                if not self.apply_text_edits(
                    uri.replace("file:///", "").replace("/", os.sep), 
                    text_edits,
                    desired_name, 
                    auto_extracted_ids=auto_extracted_ids, 
                    req_start_pos=req_start_pos, 
                    req_end_pos=req_end_pos
                ):
                    applied_all = False
        elif "documentChanges" in edit:
            for doc_change in edit["documentChanges"]:
                if "textDocument" in doc_change:
                    if not self.apply_text_edits(
                        doc_change["textDocument"]["uri"].replace("file:///", "").replace("/", os.sep),
                        doc_change["edits"], 
                        desired_name, 
                        auto_extracted_ids=auto_extracted_ids, 
                        req_start_pos=req_start_pos, 
                        req_end_pos=req_end_pos
                    ):
                        applied_all = False
        return applied_all

    
    def apply_text_edits(self, file_path, edits, desired_name, auto_extracted_ids=None, req_start_pos=None, req_end_pos=None):
        try:
            with open(file_path, 'r', encoding='utf-8', newline="") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1', newline="") as f:
                content = f.read()

        def get_absolute_offset(pos):
            lines = content.splitlines(keepends=True)
            line_idx = pos["line"]
            char_idx = pos["character"]
            return sum(len(lines[i]) for i in range(min(line_idx, len(lines)))) + char_idx

        req_start_off = get_absolute_offset(req_start_pos) if req_start_pos else None
        req_end_off = get_absolute_offset(req_end_pos) if req_end_pos else None

        # --- SOLUCIÓN DEFINITIVA A LAS MÚLTIPLES EXTRACCIONES ---
        insertions = []
        replacements = []

        # 1. Separar inserciones (nuevos métodos) de reemplazos (llamadas al método)
        for edit in edits:
            start_off = get_absolute_offset(edit["range"]["start"])
            end_off = get_absolute_offset(edit["range"]["end"])

            if start_off == end_off:
                insertions.append((start_off, end_off, edit["newText"]))
            else:
                replacements.append((start_off, end_off, edit["newText"]))

        edits_with_offsets = insertions

        # Reemplazar la sección de filtrado de replacements por una verificación de rango estricta:
        if replacements:
            if req_start_off is not None and req_end_off is not None:
                # Solo aceptamos reemplazos que pertenezcan al método actual que estamos refactorizando
                valid_replacements = [
                    rep for rep in replacements 
                    if (req_start_off - 10) <= rep[0] <= (req_end_off + 10)
                ]
                if valid_replacements:
                    best_replacement = min(valid_replacements, key=lambda rep: abs(rep[0] - req_start_off))
                    edits_with_offsets.append(best_replacement)
                if len(replacements) > len(valid_replacements):
                    print(f"      🛡️ [Escudo Activo] Se ignoraron {len(replacements) - len(valid_replacements)} reemplazos colaterales fuera del método actual.")
            else:
                edits_with_offsets.extend(replacements)

        # 3. Ordenar para aplicar de abajo hacia arriba y no desfasar los offsets
        edits_with_offsets.sort(key=lambda x: x[0], reverse=True)

        for start_off, end_off, text in edits_with_offsets:
            new_text = re.sub(r"\bextracted\d*\b", desired_name, text) if desired_name else text

            # Usar el nuevo pipeline centralizado
            new_text, applied_fixes = run_sanitization_pipeline(new_text, desired_name)

            if "void_return" in applied_fixes:
                self.save_debug_snapshot(f"3_auto_healing_DESPUÉS_return_{desired_name}.java")
                print("      🩹 [AUTO-HEALING] Return con valor en método void corregido a 'return;'.")
            
            if "orphan_variable" in applied_fixes:
                self.save_debug_snapshot(f"3_auto_healing_DESPUÉS_return_{desired_name}.java")
                print("      🩹 [AUTO-HEALING] Patrones de variable huérfana corregidos en el texto de extracción.")
                
            if "empty_return" in applied_fixes:
                self.save_debug_snapshot(f"3_auto_healing_DESPUES_return_vacio_{desired_name}.java")
                print(f"      🩹 [AUTO-HEALING] 'return;' vacío corregido inyectando valor por defecto en '{desired_name}'.")

            if not applied_fixes:
                patron_huerfana = re.search(
                    r"\b([A-Z][a-zA-Z0-9_<>,\[\]]*|int|boolean|double|float|long|short|byte|char)\s+([a-zA-Z0-9_$]+)\s*;\s*(?:return\s+[^;]+;)?\s*\}",
                    new_text)

                if patron_huerfana:
                    tipo_var = patron_huerfana.group(1)
                    nombre_var = patron_huerfana.group(2)
                    self.save_debug_snapshot(f"3_fallo_extraccion_{desired_name}.java", content)
                    print(f"      ⚠️ [EXTRACCIÓN ABORTADA] Bug de JDT (Variable huérfana: '{tipo_var} {nombre_var}').")
                    return False

            # Aplicar la edición en el texto directamente
            content = content[:start_off] + new_text + content[end_off:]

        sanitized_content, fixed_void_return_file = sanitize_jdt_void_return_bug(content)
        if fixed_void_return_file:
            content = sanitized_content

        enc = 'utf-8'
        try:
            with open(file_path, 'w', encoding=enc, newline="") as f:
                f.write(content)
        except UnicodeDecodeError:
            enc = 'latin-1'
            with open(file_path, 'w', encoding=enc, newline="") as f:
                f.write(content)

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
    # 1. Parsear AST una sola vez de forma profesional
    existing_methods = set()
    try:
        tree = javalang.parse.parse(class_content)
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            existing_methods.add(node.name)
    except Exception as e:
        pass # Si el código original falla, usamos fallback

    prepared = []
    for method_name, ext_list in methods_dict.items():
        sorted_by_size = sorted(ext_list, key=lambda e: e["range"][1] - e["range"][0])
        extraction_n = 1

        for ext in sorted_by_size:
            desired_name = f"{method_name}_extraction_{extraction_n}"

            # 2. Validación limpia contra el AST
            if existing_methods:
                while desired_name in existing_methods:
                    extraction_n += 1
                    desired_name = f"{method_name}_extraction_{extraction_n}"
                existing_methods.add(desired_name) # Lo registramos para evitar colisiones futuras
            else:
                # Fallback por si falló javalang (ej. clase con sintaxis rota inicialmente)
                while f"{desired_name}(" in class_content or f"{desired_name} (" in class_content:
                    extraction_n += 1
                    desired_name = f"{method_name}_extraction_{extraction_n}"

            prepared.append(
                {"range": ext["range"], "depth": ext["depth"], "desired_name": desired_name,
                 "method_name": method_name})

            extraction_n += 1

    prepared.sort(key=lambda x: x["range"][1] - x["range"][0])
    for ext_id, item in enumerate(prepared, start=1): item["ext_id"] = ext_id
    return prepared


def inject_markers(file_path, prepared_extractions):
    insertions = []
    for item in prepared_extractions:
        r, ext_id, depth = item["range"], item["ext_id"], item["depth"]
        size = r[1] - r[0]
        insertions.append(
            {"pos": r[1], "text": f"/*END_EXT_{ext_id}*/", "is_start": False, "depth": depth, "size": size})
        insertions.append(
            {"pos": r[0], "text": f"/*START_EXT_{ext_id}*/", "is_start": True, "depth": depth, "size": size})

    def sort_key(ins):
        # 1. Procesar desde el final del archivo hacia el principio para no desfasar offsets
        p1 = -ins["pos"]

        # 2. Si coinciden en la misma posición exacta, procesamos primero START (0) y luego END (1)
        # Como las inserciones empujan el texto anterior, esto asegura que quede: /*END_...*//*START_...*/
        p2 = 0 if ins["is_start"] else 1

        # 3. EL ARREGLO CRÍTICO DE ANIDAMIENTO:
        # - Para START: Procesamos primero los bloques PEQUEÑOS (Inner). Al insertar luego el GRANDE (Outer) en la misma posición, el GRANDE quedará por FUERA.
        # - Para END: Procesamos primero los bloques GRANDES (Outer). Al insertar luego el PEQUEÑO (Inner), el PEQUEÑO quedará por DENTRO.
        p3 = ins["size"] if ins["is_start"] else -ins["size"]

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


def sanitize_jdt_void_return_bug(text):
    """
    Corrige el patrón de JDT que, al extraer un método, puede dejar un 'return expr;'
    dentro de un método void. Si 'expr' es una llamada a función o instrucción activa,
    mantiene la llamada ('expr; return;') en lugar de eliminarla.
    """

    def find_matching_brace(source, open_brace_idx):
        depth = 0
        for idx in range(open_brace_idx, len(source)):
            char = source[idx]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return idx
        return None

    header_pattern = re.compile(r"\bvoid\b[^;{}\n]*\([^;{}]*\)\s*\{")

    updated_text = text
    changed = False
    max_passes = 8

    for _ in range(max_passes):
        cursor = 0
        parts = []
        any_changes = False

        while True:
            match = header_pattern.search(updated_text, cursor)
            if not match:
                parts.append(updated_text[cursor:])
                break

            open_brace_idx = match.end() - 1
            close_brace_idx = find_matching_brace(updated_text, open_brace_idx)
            if close_brace_idx is None:
                parts.append(updated_text[cursor:])
                break

            parts.append(updated_text[cursor:match.start()])
            body_start = open_brace_idx + 1
            body_end = close_brace_idx
            body_text = updated_text[body_start:body_end]

            # Reemplazo inteligente: preserva llamadas a métodos
            def replace_invalid_void_return(m):
                expr = m.group(1).strip()
                # Si contiene llamadas a función, asignaciones u objetos, preservamos la ejecución
                if any(c in expr for c in ['(', '=', '++', '--']) or expr.startswith("new "):
                    return f"{expr};\n        return;"
                return "return;"

            repaired_body, count = re.subn(r"\breturn\s+([^;]+);", replace_invalid_void_return, body_text)
            if count > 0:
                any_changes = True
            parts.append(updated_text[match.start():body_start] + repaired_body + "}")
            cursor = close_brace_idx + 1

        new_text = "".join(parts)

        if not any_changes:
            break

        updated_text = new_text
        changed = True

    return updated_text, changed


def sanitize_jdt_orphaned_variable_bug(text):
    """
    Intenta corregir el patrón típico de JDT al extraer métodos cuando genera una
    declaración huérfana como 'int tmp;' seguida de un return y el cierre del
    bloque. En lugar de abortar la extracción, se convierte en una declaración
    inicializada con un valor por defecto seguro.
    """

    # Regex mejorada para soportar genéricos anidados, espacios, comas y comodines (?)
    pattern = re.compile(
        r"\b([A-Z]\w*(?:\s*<[\w\s,\.\?<>\[\]]+>)?(?:\[\])*|int|boolean|double|float|long|short|byte|char(?:\[\])*)\s+([a-zA-Z0-9_$]+)\s*;(?=\s*(?:return\s+[^;]+;)?\s*\})",
        re.MULTILINE,
    )

    def replace_orphan(match):
        var_type = match.group(1)
        var_name = match.group(2)

        if var_type in ["double", "float"]:
            default_value = "0.0"
        elif var_type in ["int", "long", "short", "byte", "char"]:
            default_value = "0"
        elif var_type == "boolean":
            default_value = "false"
        else:
            default_value = "null"

        return f"{var_type} {var_name} = {default_value};"

    new_text, count = pattern.subn(replace_orphan, text)
    return new_text, count > 0


def sanitize_uninitialized_variables(content):
    """
    Descompone declaraciones múltiples (ej. double y1, y2, beta;) en líneas
    independientes e inicializadas para evitar que el refactorizador de JDT LS
    borre variables del ámbito padre.
    """

    def split_and_initialize(match):
        indent = match.group(1) or ""
        var_type = match.group(2)
        vars_payload = match.group(3)
        comment = match.group(4) or ""

        # Determinar valor por defecto según el tipo primitivo/objeto
        if var_type in ["double", "float"]:
            default_val = "0.0"
        elif var_type in ["int", "long", "short", "byte", "char"]:
            default_val = "0"
        elif var_type == "boolean":
            default_val = "false"
        else:
            default_val = "null"

        split_lines = []
        # Separar las variables por coma
        for var_expr in vars_payload.split(","):
            var_clean = var_expr.strip()
            if not var_clean:
                continue
            # Si ya tiene asignación previa (ej. x = 5), se conserva; si no, se inicializa
            if "=" in var_clean:
                split_lines.append(f"{indent}{var_type} {var_clean};")
            else:
                split_lines.append(f"{indent}{var_type} {var_clean} = {default_val};")

        result_code = "\n".join(split_lines)
        if comment:
            result_code += f" {comment}"
        return result_code

    # Regex que detecta declaraciones simples Y múltiples (con comas)
    # Regex mejorada para soportar genéricos complejos en declaraciones múltiples
    multidecl_regex = r'^(\s*)([A-Z]\w*(?:\s*<[\w\s,\.\?<>\[\]]+>)?(?:\[\])*|int|double|float|long|short|byte|char|boolean(?:\[\])*)\s+([a-zA-Z0-9_$\s,=]+);\s*(/\*.*?\*/)?\s*$'

    return re.sub(multidecl_regex, split_and_initialize, content, flags=re.MULTILINE)


def run_sanitization_pipeline(text, desired_name):
    """
    Aplica todas las correcciones conocidas de JDT LS en orden estricto de precedencia.
    Devuelve el texto modificado y una lista de las curaciones aplicadas.
    """
    applied_fixes = []
    
    # 1. Variables no inicializadas globales
    text = sanitize_uninitialized_variables(text)
    
    # 2. Retornos vacíos en métodos no-void
    text, fixed_empty = sanitize_jdt_empty_return_in_non_void(text, desired_name)
    if fixed_empty: applied_fixes.append("empty_return")

    # 3. Bug de 'return expr;' dentro de métodos void
    text, fixed_void = sanitize_jdt_void_return_bug(text)
    if fixed_void: applied_fixes.append("void_return")

    # 4. Variables huérfanas
    text, fixed_orphan = sanitize_jdt_orphaned_variable_bug(text)
    if fixed_orphan: applied_fixes.append("orphan_variable")

    return text, applied_fixes


def sanitize_jdt_empty_return_in_non_void(text, desired_name):
    """
    Detecta si el método extraído (no-void) contiene un 'return;' vacío 
    provocado por un 'break' o 'continue' mal traducido por JDT LS, 
    y le inyecta un valor por defecto.
    """
    # 1. Buscar la firma del método extraído para averiguar su tipo de retorno
    patron_firma = r"\b(?:public\s+|private\s+|protected\s+|static\s+)*([A-Z]\w*(?:\s*<[\w\s,\.\?<>\[\]]+>)?(?:\[\])*|(?:int|double|float|long|short|byte|char|boolean)(?:\[\])*)\s+" + re.escape(desired_name) + r"\s*\("
    
    match = re.search(patron_firma, text)
    if not match:
        return text, False

    tipo_retorno = match.group(1)

    # Si es void, no hacemos nada (tu otra función ya se encarga de los void)
    if tipo_retorno == "void":
        return text, False

    # 2. Determinar el valor por defecto seguro según el tipo
    if "[]" in tipo_retorno: 
        def_val = "null"
    else:
        if tipo_retorno in ["double", "float"]:
            def_val = "0.0"
        elif tipo_retorno in ["int", "long", "short", "byte", "char"]:
            def_val = "0"
        elif tipo_retorno == "boolean":
            def_val = "false"
        else:
            def_val = "null"

    # 3. Reemplazar 'return;' literal por el retorno tipado
    repaired_text, count = re.subn(r"\breturn\s*;", f"return {def_val};", text)

    return repaired_text, count > 0


# =====================================================================
# MOTOR PRINCIPAL DE REFACTORIZACIÓN CON SEGUIMIENTO DE MÉTRICAS
# =====================================================================
def apply_refactorings_to_classes(class_offsets_map, project_root, use_system_m2=False, debug_dir=None):
    temp_ws = tempfile.mkdtemp(prefix="jdtls_ws_")
    client = JDTLSClient(JDTLS_HOME, temp_ws, project_root, use_system_m2=use_system_m2)
    extraction_results = []

    try:
        client.start()
        client.initialize()

        total_classes = len(class_offsets_map)
        for index, (class_rel_path, methods_dict) in enumerate(class_offsets_map.items(), 1):
            if not methods_dict: continue

            # 1. Intentar la ruta relativa completa
            full_class_path = os.path.join(project_root, class_rel_path)

            # 2. Si no existe, buscar validando la ruta (paquetes) para no mezclar clases con el mismo nombre
            if not os.path.exists(full_class_path):
                found_path = None
                target_filename = os.path.basename(class_rel_path)

                # Extraemos las últimas 3 partes de la ruta para diferenciar paquetes (ej: nsgaiii/util/EnvironmentalSelection.java)
                path_parts = class_rel_path.replace("\\", "/").split("/")
                suffix_to_match = "/".join(path_parts[-3:]) if len(path_parts) >= 3 else class_rel_path

                for current_dir, dirs, files in os.walk(project_root):
                    if target_filename in files:
                        potential_path = os.path.join(current_dir, target_filename).replace("\\", "/")
                        if potential_path.endswith(suffix_to_match):
                            found_path = os.path.normpath(potential_path)
                            break

                if found_path:
                    full_class_path = found_path
                else:
                    print(
                        f"   👻 Archivo fantasma: No se encontró '{class_rel_path}' ({target_filename}) en {os.path.basename(project_root)}")
                    continue

            porcentaje_progreso = (index / total_classes) * 100
            print(
                f"\n📁 Procesando clase [{index}/{total_classes}] ({porcentaje_progreso:.1f}% completado): {os.path.basename(full_class_path)}")

            # Preparar variables de debug (pero NO crear la carpeta todavía)
            class_debug_dir = None
            debug_initialized = False
            if debug_dir:
                class_name = os.path.basename(full_class_path).replace(".java", "")
                class_debug_dir = os.path.join(debug_dir, class_name)

            # Auxiliar para volcar fotos EXACTAS del archivo en disco en cualquier momento
            def save_debug_snapshot(filename, code_to_save=None):
                nonlocal debug_initialized
                if not class_debug_dir:
                    return

                if not debug_initialized:
                    os.makedirs(class_debug_dir, exist_ok=True)
                    with open(os.path.join(class_debug_dir, "1_original.java"), "w", encoding=encoding_usado, newline="") as f:
                        f.write(class_content)
                    with open(os.path.join(class_debug_dir, "2_con_marcadores.java"), "w", encoding=encoding_usado, newline="") as f:
                        f.write(marked_content)
                    debug_initialized = True

                # Si no nos pasan código explícito, leemos la FOTO REAL de la clase en disco
                if code_to_save is None:
                    try:
                        with open(full_class_path, 'r', encoding=encoding_usado, newline="") as f:
                            code_to_save = f.read()
                    except Exception:
                        code_to_save = ""

                with open(os.path.join(class_debug_dir, filename), "w", encoding=encoding_usado, newline="") as f:
                    f.write(code_to_save)

            client.save_debug_snapshot = save_debug_snapshot

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

            # --- 4. PREPROCESAMIENTO SEGURO  ---
            # Aplicamos el saneamiento de variables sobre el código YA MARCADO.
            # Al hacerlo ahora, el cambio de longitud del archivo no desalinea los offsets originales.
            # marked_content = pre_process_java_file(marked_content)
            marked_content = sanitize_uninitialized_variables(marked_content)

            with open(full_class_path, "w", encoding=encoding_usado, newline="") as f:
                f.write(marked_content)

            time.sleep(3)
            client.open_file(full_class_path)

            class_auto_extracted = set()
            last_processed_method = None  # Seguimiento del método actual

            for ext_item in prepared_extractions:
                ext_id, desired_name = ext_item["ext_id"], ext_item["desired_name"]
                current_method_name = ext_item["method_name"]

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

                if start_idx == -1 or end_idx == -1:
                    print(f"\n      ⚠️ Marcadores de '{desired_name}' no encontrados. Saltando...")

                    # Volcar debug solo cuando hay un fallo (On-Demand)
                    save_debug_snapshot(f"3_fallo_sin_marcadores_{desired_name}.java", content)

                    continue

                raw_code_snippet = content[start_idx + len(start_marker): end_idx].strip()
                snippet_lines = raw_code_snippet.splitlines()
                preview_text = snippet_lines[0] if snippet_lines else ""
                if len(preview_text) > 80: preview_text = preview_text[:77] + "..."

                # --- LIMPIEZA DE MARCADORES Y CÁLCULO PRECISO DE OFFSETS ---
                S_len = len(start_marker)
                E_len = len(end_marker)

                # 1. Extraer el fragmento exacto entre los marcadores
                snippet = content[start_idx + S_len: end_idx]

                # 2. Eliminar cualquier otro marcador residual dentro del fragmento (ej. marcadores de extracciones fallidas)
                snippet_clean = re.sub(r'/\*(START|END)_EXT_\d+\*/', '', snippet)

                # 3. Reconstruir el archivo sin el marcador actual y con el fragmento sin basura
                content = content[:start_idx] + snippet_clean + content[end_idx + E_len:]

                # 4. Ajustar bordes de selección para omitir saltos de línea y espacios en blanco
                sel_start = start_idx
                sel_end = start_idx + len(snippet_clean)

                while sel_start < sel_end and content[sel_start].isspace():
                    sel_start += 1

                while sel_end > sel_start and content[sel_end - 1].isspace():
                    sel_end -= 1

                # 5. Guardar en disco el contenido limpio para que Eclipse analice el código Java real
                with open(full_class_path, "w", encoding=encoding_usado, newline="") as f:
                    f.write(content)

                client.clear_file_errors(full_class_path)

                # 6. Convertir los offsets limpios a coordenadas (línea, columna)
                start_pos = offset_to_position(content, sel_start)
                end_pos = offset_to_position(content, sel_end)

                print(f"\n👉 [Extracción {ext_id}/{len(prepared_extractions)}] Asignando: '{desired_name}'")
                print(f"   📝 Código: \"{preview_text}\"")

                client.send("textDocument/didClose",
                            {"textDocument": {"uri": f"file:///{full_class_path.replace('\\', '/')}"}},
                            is_notification=True)

                # --- ENVIAMOS A ECLIPSE EL CÓDIGO REAL ---
                # Eclipse sabe ignorar comentarios /*START...*/ sin romper el AST.
                client.send("textDocument/didOpen", {
                    "textDocument": {"uri": f"file:///{full_class_path.replace('\\', '/')}", "languageId": "java",
                                     "version": int(time.time()), "text": content}
                }, is_notification=True)

                success = False
                max_intentos = 2  # 1 intento original + 1 reintento tras recuperación
                error_line_number = 0

                # --- Set temporal para evitar falsos positivos en caso de rollback ---
                temp_auto_extracted = set()

                # save_debug_snapshot(f"01_PRE_extraccion_{desired_name}.java", content)

                for intento in range(1, max_intentos + 1):
                    try:
                        success = client.request_extract_method(
                            full_class_path, start_pos, end_pos, desired_name,
                            timeout=100, auto_extracted_ids=temp_auto_extracted
                        )
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

                    # save_debug_snapshot(f"02_POST_extraccion_{desired_name}.java", new_content)

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

                        # --- Consolidar los auto-extraídos porque todo ha ido bien ---
                        class_auto_extracted.update(temp_auto_extracted)

                        extraction_results.append(
                            {"Clase": os.path.basename(full_class_path), "Metodo Original": ext_item["method_name"],
                             "Nombre Extraccion": desired_name, "Exito": "Sí"})
                    else:
                        print(f"      🚨 La extracción generó {len(errors)} error(es) de compilación en Eclipse.")
                        print(f"         ↳ Detalle: {errors[0].get('message')}")
                        success = False  # Asumimos fallo inicialmente

                        # Extraemos la línea de las coordenadas JSON, no del texto del mensaje
                        error_line_number = errors[0].get('range', {}).get('start', {}).get('line', 0) + 1

                        # AUTO-HEALING: VARIABLES NO RESUELTAS (Falta declaración de tipo)
                        if error_line_number and "cannot be resolved to a variable" in errors[0].get('message'):
                            heal_attempts = 0
                            max_heal_attempts = 3 * len(errors)
                            current_errors = errors

                            while current_errors and "cannot be resolved to a variable" in current_errors[0].get(
                                    'message') and heal_attempts < max_heal_attempts:
                                # 1. Extraer el nombre de la variable
                                match = re.search(r"(\w+)\s+cannot be resolved to a variable",
                                                  current_errors[0].get('message'))
                                if match:
                                    missing_var = match.group(1)
                                    error_line = current_errors[0].get('range', {}).get('start', {}).get('line', 0) + 1

                                    # 2. 🔍 BUSCAR EL TIPO REAL EN EL BACKUP DEL CÓDIGO ORIGINAL
                                    real_type = "var"
                                    # Tipos Java vitaminados para capturar CUALQUIER genérico complejo
                                    tipos_java = r"([A-Z]\w*(?:\s*<[\w\s,\.\?<>\[\]]+>)?(?:\[\])*|int|double|float|long|short|byte|char|boolean(?:\[\])*)"
                                    regex_tipo = r"(?:public\s+|private\s+|protected\s+|final\s+|static\s+)*" + tipos_java + r"\s+[a-zA-Z0-9_\s,]*\b" + re.escape(missing_var) + r"\b"

                                    for line in backup_content_clean.splitlines():
                                        match_tipo = re.search(regex_tipo, line.strip())
                                        if match_tipo and "=" not in line.split(missing_var)[0]:
                                            real_type = match_tipo.group(1)
                                            break

                                    # 3. Leer el archivo Java
                                    with open(full_class_path, 'r', encoding=encoding_usado, newline='') as f:
                                        lines = f.readlines()

                                    # 4. Buscar la línea exacta que está fallando
                                    idx = error_line - 1

                                    if idx < len(lines):
                                        problematic_line = lines[idx]

                                        # 5. Inyectar el tipo dinámico
                                        if missing_var in problematic_line:
                                            print(
                                                f"      🩹 [AUTO-HEALING] Declarando '{missing_var}' in-situ con su tipo original: '{real_type}' (Intento {heal_attempts + 1})...")

                                            fixed_line = re.sub(rf"\b{missing_var}\s*=", f"{real_type} {missing_var} =",
                                                                problematic_line, count=1)
                                            lines[idx] = fixed_line

                                            # 6. Sobrescribir el archivo curado en disco
                                            with open(full_class_path, 'w', encoding=encoding_usado, newline='') as f:
                                                f.writelines(lines)

                                            # Sincronizar
                                            current_errors = client.sync_file_and_get_errors(full_class_path, timeout=5.0)
                                            heal_attempts += 1
                                        else:
                                            # Si por algún motivo no encontramos la variable en la línea, salimos del bucle
                                            break
                                else:
                                    break

                            # Evaluación final tras salir del bucle
                            # Leer el estado del código curado tras el proceso de healing
                            try:
                                with open(full_class_path, 'r', encoding=encoding_usado, newline='') as f:
                                    healed_code = f.read()
                            except Exception:
                                healed_code = content

                            # Evaluación final tras salir del bucle
                            if not current_errors:
                                print(
                                    "      ✨ [AUTO-HEALING EXITOSO] Todos los errores de resolución de variables curados.")
                                save_debug_snapshot(f"autohealing_exito_{desired_name}.java", healed_code)
                                print("      ✅ ¡Extracción exitosa!")
                                success = True
                                extraction_results.append({
                                    "Clase": os.path.basename(full_class_path),
                                    "Metodo Original": ext_item["method_name"],
                                    "Nombre Extraccion": desired_name,
                                    "Exito": "Sí (Auto-curado)"
                                })
                                # Actualizamos el backup para las siguientes extracciones
                                backup_content_clean = healed_code
                            else:
                                print(
                                    "      ⚠️ [AUTO-HEALING FALLIDO] Persisten otros errores tras los intentos de curación.")
                                save_debug_snapshot(f"autohealing_fallo_{desired_name}.java", healed_code)
                                for err in current_errors:
                                    err_line = err.get('range', {}).get('start', {}).get('line', 0) + 1
                                    print(f"         ↳ [Línea {err_line}]: {err.get('message')}")

                        # --- NUEVO AUTO-HEALING: VARIABLES NO INICIALIZADAS ---
                        elif error_line_number and "may not have been initialized" in errors[0].get('message'):
                            heal_attempts = 0
                            max_heal_attempts = 3 * len(errors)
                            current_errors = errors

                            while current_errors and "may not have been initialized" in current_errors[0].get(
                                    'message') and heal_attempts < max_heal_attempts:
                                match = re.search(r"The local variable (\w+) may not have been initialized",
                                                  current_errors[0].get('message'))
                                if match:
                                    uninit_var = match.group(1)
                                    print(
                                        f"      🩹 [AUTO-HEALING] Inicializando variable huérfana '{uninit_var}' a null (Intento {heal_attempts + 1})...")

                                    # Leer y parchear el archivo
                                    with open(full_class_path, 'r', encoding=encoding_usado, newline='') as f:
                                        lines = f.readlines()

                                    for j, line in enumerate(lines):
                                        if re.search(rf"\b{uninit_var}\s*;", line):
                                            match_tipo = re.search(r"([A-Z]\w*(?:\s*<[\w\s,\.\?<>\[\]]+>)?(?:\[\])*|int|double|float|long|short|byte|char|boolean(?:\[\])*)\s+" + re.escape(uninit_var) + r"\s*;", line)
                                            if match_tipo:
                                                var_type = match_tipo.group(1)
                                                if var_type in ["double", "float"]: def_val = "0.0"
                                                elif var_type in ["int", "long", "short", "byte", "char"]: def_val = "0"
                                                elif var_type == "boolean": def_val = "false"
                                                else: def_val = "null"
                                                lines[j] = re.sub(rf"\b({uninit_var})\s*;", rf"\1 = {def_val};", line)
                                            break

                                    with open(full_class_path, 'w', encoding=encoding_usado, newline='') as f:
                                        f.writelines(lines)

                                    # Sincronizar
                                    current_errors = client.sync_file_and_get_errors(full_class_path, timeout=5.0)
                                    heal_attempts += 1
                                else:
                                    break

                            # Leer el estado del código curado tras el proceso de healing
                            try:
                                with open(full_class_path, 'r', encoding=encoding_usado, newline='') as f:
                                    healed_code = f.read()
                            except Exception:
                                healed_code = content

                            if not current_errors:
                                print("      ✨ [AUTO-HEALING EXITOSO] Todos los errores de inicialización resueltos.")
                                save_debug_snapshot(f"autohealing_exito_{desired_name}.java", healed_code)
                                print("      ✅ ¡Extracción exitosa!")
                                success = True
                                extraction_results.append({
                                    "Clase": os.path.basename(full_class_path),
                                    "Metodo Original": ext_item["method_name"],
                                    "Nombre Extraccion": desired_name,
                                    "Exito": "Sí (Auto-curado)"
                                })
                                backup_content_clean = healed_code
                            else:
                                print(
                                    "      ⚠️ [AUTO-HEALING FALLIDO] Persisten otros errores tras los intentos de curación.")
                                save_debug_snapshot(f"autohealing_fallo_{desired_name}.java", healed_code)

                        # --- AUTO-HEALING: TIPO GENÉRICO PERDIDO (MISSING TYPE) ---
                        elif error_line_number and "refers to the missing type" in errors[0].get('message'):
                            heal_attempts = 0
                            max_heal_attempts = 3 * len(errors)
                            current_errors = errors

                            while current_errors and "refers to the missing type" in current_errors[0].get('message') and heal_attempts < max_heal_attempts:
                                # 1. Extraer el MÉTODO AFECTADO y el TIPO GENÉRICO desde el mensaje de error
                                error_msg = current_errors[0].get('message')
                                match_err = re.search(r"The method\s+(\w+)\b.*?refers to the missing type\s+(\w+)", error_msg)
                                
                                if match_err:
                                    target_method = match_err.group(1)
                                    missing_type = match_err.group(2)
                                else:
                                    # Fallback por si el mensaje tiene otro formato
                                    match_fallback = re.search(r"refers to the missing type\s+(\w+)", error_msg)
                                    target_method = desired_name
                                    missing_type = match_fallback.group(1) if match_fallback else None

                                if missing_type:
                                    print(f"      🩹 [AUTO-HEALING] Inyectando tipo genérico '<{missing_type}>' en la firma de '{target_method}' (Intento {heal_attempts + 1})...")

                                    # 2. Pedir el AST a Eclipse LSP para encontrar la línea EXACTA de la declaración
                                    symbols = client.request_document_symbols(full_class_path)
                                    decl_line = client.find_method_line_in_symbols(symbols, target_method)

                                    with open(full_class_path, 'r', encoding=encoding_usado, newline='') as f:
                                        lines = f.readlines()

                                    # --- NUEVO: FALLBACK TEXTUAL DE ALTA PRECISIÓN ---
                                    # Si Eclipse falló al generar el AST porque la clase está muy rota, 
                                    # buscamos la firma del método recién extraído de forma manual.
                                    if decl_line is None:
                                        for idx, line_text in enumerate(lines):
                                            if re.search(rf"\b(?:public|private|protected)\b.*?\b{re.escape(target_method)}\s*\(", line_text):
                                                decl_line = idx
                                                break
                                    # ------------------------------------------------

                                    # Si encontramos la declaración (vía AST o vía texto)
                                    if decl_line is not None and decl_line < len(lines):
                                        line = lines[decl_line]
                                        
                                        # Asegurarnos de no inyectarlo si ya está presente
                                        if f"<{missing_type}>" not in line:
                                            # Ahora que estamos 100% seguros de que ESTA es la firma, inyectamos el genérico
                                            if "private static " in line:
                                                lines[decl_line] = line.replace("private static ", f"private static <{missing_type}> ", 1)
                                            elif "private " in line:
                                                lines[decl_line] = line.replace("private ", f"private <{missing_type}> ", 1)
                                            elif "protected static " in line:
                                                lines[decl_line] = line.replace("protected static ", f"protected static <{missing_type}> ", 1)
                                            elif "protected " in line:
                                                lines[decl_line] = line.replace("protected ", f"protected <{missing_type}> ", 1)
                                            elif "public static " in line:
                                                lines[decl_line] = line.replace("public static ", f"public static <{missing_type}> ", 1)
                                            elif "public " in line:
                                                lines[decl_line] = line.replace("public ", f"public <{missing_type}> ", 1)
                                            else:
                                                lines[decl_line] = re.sub(rf"(\b{re.escape(target_method)}\s*\()", f"<{missing_type}> \\1", line, count=1)
                                    else:
                                        print(f"      ⚠️ No se encontró la declaración de '{target_method}' en el AST. Abortando inyección de genérico.")
                                        break

                                    # 3. Guardar el archivo curado
                                    with open(full_class_path, 'w', encoding=encoding_usado, newline='') as f:
                                        f.writelines(lines)

                                    # Sincronizar
                                    current_errors = client.sync_file_and_get_errors(full_class_path, timeout=5.0)
                                    heal_attempts += 1
                                else:
                                    break

                            # Evaluación final tras el bucle de curación
                            try:
                                with open(full_class_path, 'r', encoding=encoding_usado, newline='') as f:
                                    healed_code = f.read()
                            except Exception:
                                healed_code = content

                            if not current_errors:
                                print("      ✨ [AUTO-HEALING EXITOSO] Error de tipo genérico resuelto.")
                                save_debug_snapshot(f"autohealing_exito_{desired_name}.java", healed_code)
                                print("      ✅ ¡Extracción exitosa!")
                                success = True
                                extraction_results.append({
                                    "Clase": os.path.basename(full_class_path),
                                    "Metodo Original": ext_item["method_name"],
                                    "Nombre Extraccion": desired_name,
                                    "Exito": "Sí (Auto-curado genéricos)"
                                })
                                backup_content_clean = healed_code
                            else:
                                print("      ⚠️ [AUTO-HEALING FALLIDO] Persisten otros errores tras intentar inyectar el genérico.")
                                save_debug_snapshot(f"autohealing_fallo_{desired_name}.java", healed_code)
                        
                        # --- NUEVO AUTO-HEALING: RETURN FALTANTE ---
                        elif error_line_number and "This method must return a result of type" in errors[0].get('message'):
                            heal_attempts = 0
                            max_heal_attempts = 3 * len(errors)
                            current_errors = errors

                            while current_errors and "This method must return a result of type" in current_errors[0].get('message') and heal_attempts < max_heal_attempts:
                                match = re.search(r"This method must return a result of type\s+([^\s]+)", current_errors[0].get('message'))
                                if match:
                                    original_type = match.group(1)
                                    print(f"      🩹 [AUTO-HEALING] Inyectando return por defecto para el tipo '{original_type}' en la línea {error_line_number} (Intento {heal_attempts + 1})...")

                                    # Determinar el valor por defecto seguro
                                    if "[]" in original_type: 
                                        def_val = "null" 
                                    else:
                                        if original_type in ["double", "float"]: def_val = "0.0"
                                        elif original_type in ["int", "long", "short", "byte", "char"]: def_val = "0"
                                        elif original_type == "boolean": def_val = "false"
                                        else: def_val = "null"

                                    with open(full_class_path, 'r', encoding=encoding_usado, newline='') as f:
                                        lines = f.readlines()
                                    
                                    # Índice de la línea que Eclipse marca como errónea (0-indexed)
                                    idx = error_line_number - 1
                                    injected = False
                                    
                                    if 0 <= idx < len(lines):
                                        target_line = lines[idx]
                                        
                                        # Caso 1: Eclipse marca un "return;" vacío (como en tu ejemplo de la línea 89)
                                        if re.search(r"\breturn\s*;", target_line):
                                            lines[idx] = re.sub(r"\breturn\s*;", f"return {def_val};", target_line)
                                            injected = True
                                        
                                        # Caso 2: Eclipse marca la llave de cierre '}' porque falta el return al final
                                        elif "}" in target_line:
                                            # Insertamos el return justo antes de la llave respetando la indentación
                                            indent = target_line[:len(target_line) - len(target_line.lstrip())]
                                            lines[idx] = target_line.replace("}", f"return {def_val};\n{indent}", 1)
                                            injected = True
                                        
                                        # Caso 3: Fallback (Eclipse marca el inicio del método por falta de return al final)
                                        else:
                                            print(f"      ⚠️ [AUTO-HEALING] Eclipse marcó la instrucción '{target_line.strip()}'. Buscando el final EXACTO del método mediante balanceo de llaves...")
                                            
                                            injected = False
                                            bracket_count = 0
                                            found_first_bracket = False
                                            method_end_idx = -1
                                            
                                            # Recorrer desde la línea del error hacia abajo para balancear llaves
                                            for j in range(idx, len(lines)):
                                                line_text = lines[j]
                                                
                                                # Contamos las aperturas y cierres iterando por carácter
                                                for char in line_text:
                                                    if char == '{':
                                                        bracket_count += 1
                                                        found_first_bracket = True
                                                    elif char == '}':
                                                        bracket_count -= 1
                                                
                                                # Si ya abrimos la primera llave y el contador vuelve a 0, es el cierre del método
                                                if found_first_bracket and bracket_count == 0:
                                                    method_end_idx = j
                                                    break
                                                    
                                            if method_end_idx != -1:
                                                j = method_end_idx
                                                indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                                                
                                                # Inyectamos el return justo antes de la última llave de cierre
                                                partes = lines[j].rsplit('}', 1)
                                                lines[j] = partes[0] + f"return {def_val};\n{indent}}}" + (partes[1] if len(partes) > 1 else "")
                                                
                                                injected = True
                                                print(f"      🩹 [AUTO-HEALING] Return inyectado en el final EXACTO del bloque (Línea {j+1}) evitando 'Unreachable code'.")
                                            else:
                                                print("      ⚠️ [AUTO-HEALING] Abortando: No se pudo balancear las llaves para encontrar el final del método.")
                                                break

                                    if injected:
                                        # Escribir los cambios en disco
                                        with open(full_class_path, 'w', encoding=encoding_usado, newline='') as f:
                                            f.writelines(lines)
                                            
                                        # Sincronizar
                                        current_errors = client.sync_file_and_get_errors(full_class_path, timeout=5.0)
                                        
                                        # 🔥 CRÍTICO: Actualizar el número de línea por si el error saltó a otro "return;" vacío en el mismo método
                                        if current_errors:
                                            error_line_number = current_errors[0].get('range', {}).get('start', {}).get('line', 0) + 1
                                            
                                        heal_attempts += 1
                                    else:
                                        print("      ⚠️ [AUTO-HEALING] No se pudo inyectar el return en la línea indicada.")
                                        break
                                else:
                                    break
                                # Evaluación final tras el bucle de curación de returns
                                try:
                                    with open(full_class_path, 'r', encoding=encoding_usado, newline='') as f:
                                        healed_code = f.read()
                                except Exception:
                                    healed_code = content

                                if not current_errors:
                                    print("      ✨ [AUTO-HEALING EXITOSO] Todos los returns faltantes inyectados de forma segura.")
                                    save_debug_snapshot(f"autohealing_exito_{desired_name}.java", healed_code)
                                    print("      ✅ ¡Extracción exitosa!")
                                    success = True
                                    extraction_results.append({
                                        "Clase": os.path.basename(full_class_path),
                                        "Metodo Original": ext_item["method_name"],
                                        "Nombre Extraccion": desired_name,
                                        "Exito": "Sí (Auto-curado return)"
                                    })
                                    backup_content_clean = healed_code
                                else:
                                    print("      ⚠️ [AUTO-HEALING FALLIDO] Persisten otros errores tras intentar inyectar los returns.")
                                    save_debug_snapshot(f"autohealing_fallo_{desired_name}.java", healed_code)

                if not success:
                    extraction_results.append({
                        "Clase": os.path.basename(full_class_path),
                        "Metodo Original": ext_item["method_name"],
                        "Nombre Extraccion": desired_name,
                        "Exito": "No"
                    })

                    # Volcar el archivo completo a la carpeta de debug
                    codigo_a_guardar = failed_code if 'failed_code' in locals() else backup_content_clean
                    save_debug_snapshot(f"3_fallo_extraccion_{desired_name}.java", codigo_a_guardar)

                    print("\n      ❌ Deshaciendo extracción (Rollback) para mantener la clase funcional...")
                    with open(full_class_path, "w", encoding=encoding_usado, newline="") as f:
                        f.write(backup_content_clean)

            client.close_file(full_class_path)

            # Guardar estado final SOLO si se activó el debug por algún error
            try:
                with open(full_class_path, 'r', encoding='utf-8', newline="") as f:
                    final_content = f.read()
            except UnicodeDecodeError:
                with open(full_class_path, 'r', encoding='latin-1', newline="") as f:
                    final_content = f.read()
            save_debug_snapshot("4_refactorizada_final.java", final_content)

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
    
    # =====================================================================
    # --- INICIALIZACIÓN DEL LOG DINÁMICO ---
    # =====================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.target_class:
        # Limpiamos el nombre de la clase por si acaso trae rutas ("paquete/Clase.java")
        clean_class = args.target_class.replace("/", "_").replace("\\", "_").replace(".java", "")
        log_filename = f"{project_name}_{clean_class}.log"
    else: # Si se marca --all
        log_filename = f"{project_name}_all.log"

    log_file_path = os.path.join(script_dir, log_filename)
    
    # Aplicar la redirección
    sys.stdout = DualLogger(log_file_path)
    sys.stderr = sys.stdout
    # =====================================================================

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

    # --- Calcular el total de extracciones marcadas en los CSV ---
    total_planificadas = sum(len(exts) for methods in class_map.values() for exts in methods.values())

    print("\n--- INICIANDO REFACTORIZACIÓN EN LOTE ---")
    start_time_global = time.time()

    # Crear directorio base para depuración al mismo nivel que el proyecto
    parent_dir = os.path.dirname(os.path.normpath(args.project_root))
    debug_base_dir = os.path.join(parent_dir, f"{project_name}_debug_classes")
    os.makedirs(debug_base_dir, exist_ok=True)

    extraction_results = apply_refactorings_to_classes(class_map, project_root,
                                                       use_system_m2=args.use_system_m2, debug_dir=debug_base_dir)

    total_intentos = len(extraction_results)

    if total_planificadas > 0:
        exitosos = sum(1 for r in extraction_results if r["Exito"].startswith("Sí"))
        fallidos = total_intentos - exitosos
        omitidas = total_planificadas - total_intentos
        porcentaje = (exitosos / total_planificadas) * 100

        parent_dir = os.path.dirname(os.path.normpath(args.project_root))
        csv_filename = f"{project_name}_metricas_extraccion.csv"
        csv_path = os.path.abspath(os.path.join(parent_dir, csv_filename))

        df_results = pd.DataFrame(extraction_results)
        df_results.to_csv(csv_path, index=False, encoding="utf-8")

        # --- AGRUPACIÓN A NIVEL DE MÉTODO ---
        method_stats = []
        for (clase, metodo), group in df_results.groupby(["Clase", "Metodo Original"]):
            total_ext = len(group)
            exitos = sum(group["Exito"].str.startswith("Sí"))
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

        total_time_seconds = time.time() - start_time_global
        m, s = divmod(total_time_seconds, 60)
        h, m = divmod(m, 60)
        time_str = f"{int(h)}h {int(m)}m {int(s)}s" if h > 0 else f"{int(m)}m {int(s)}s"

        print(f"\n⏱️ TIEMPO TOTAL DE EJECUCIÓN: {time_str}")

        print("\n📊 RENDIMIENTO POR EXTRACCIONES:")
        print(f"   * Total Planificadas (encontradas al inicio): {total_planificadas}")
        print(f"   * Total Intentadas en Eclipse: {total_intentos}")
        print(f"   * Omitidas sin llegar a probarse (ej. error de marcadores): {omitidas}")
        print(f"   * Fallidas tras probarse: {fallidos}")
        print(f"   * Exitosas: {exitosos}")
        print(f"   * Tasa de Éxito Global (sobre las planificadas): **{porcentaje:.2f}%**, es decir, {exitosos} de {total_planificadas}.")

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