"""Loopback-only UI and API; serves explicit assets, never the project directory."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from .assistant import AssistantError, Investigator, Settings
from .assistant_tools import GraphTools
from .reviews import ReviewStore, ReviewError


def create_server(data, report, out, host="127.0.0.1", port=8520, env_file=".env", investigator_factory=None):
    if host != "127.0.0.1":
        raise ValueError("MVP запускается только на 127.0.0.1.")
    graph = GraphTools(data, report)
    report_dir = Path(out)
    dataset_hashes = data.meta.get("input_sha256") or {
        "records": hashlib.sha256(json.dumps(
            [list(data.nodes.values()), data.edges, data.transactions],
            sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()}
    reviews = ReviewStore(report_dir / "reviews.jsonl", dataset_hashes, [str(gid) for gid in data.nodes])
    gate = threading.BoundedSemaphore(1)
    factory = investigator_factory or (lambda: Investigator(graph, Settings.load(env_file)))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # Questions, GIDs, and provider responses do not enter console logs.

        def allowed(self, post=False):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            current = self.headers.get("Host", "")
            if current not in hosts:
                self.reply(403, {"error": "Недопустимый адрес локального сервера.", "code": "host"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + current:
                self.reply(403, {"error": "Запрос разрешён только из интерфейса приложения.", "code": "origin"})
                return False
            if post and self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.reply(403, {"error": "Межсайтовый запрос отклонён.", "code": "origin"})
                return False
            return True

        def send_bytes(self, status, body, content_type, attachment=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def reply(self, status, obj):
            self.send_bytes(status, json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if not self.allowed():
                return
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            try:
                if parts.path in ("/", "/assistant"):
                    self.send_bytes(200, Path(__file__).with_name("assistant.html").read_bytes(), "text/html; charset=utf-8")
                elif parts.path in ("/dashboard", "/dashboard.html"):
                    self.send_bytes(200, (report_dir / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
                elif parts.path == "/api/status":
                    self.reply(200, factory().status())
                elif parts.path == "/api/nodes":
                    limit = max(1, min(10000, int(query.get("limit", ["20"])[0])))
                    ranked = sorted(report["nodes"], key=lambda n: (-n["priority_score"], int(n["gid"])))
                    keys = ("gid", "role", "priority_score", "role_score", "cluster_id", "evidence", "depth", "is_seed", "metrics", "next_request")
                    rows = [{key: n[key] for key in keys} for n in ranked[:limit]]
                    self.reply(200, dict(nodes=rows, total=len(ranked), returned=len(rows), meta=report["meta"]))
                elif parts.path == "/api/node":
                    result = graph.call("node_details", {"gid": query.get("gid", [""])[0]})
                    self.reply(200 if result.get("ok") else 404, result)
                elif parts.path == "/api/reviews":
                    self.reply(200, reviews.list())
                elif parts.path.startswith("/exports/"):
                    name = parts.path.removeprefix("/exports/")
                    allowed = {"nodes_roles.csv", "clusters.csv", "top_nodes.csv", "next_queries.csv", "features.csv", "stability.csv"}
                    if name not in allowed or not (report_dir / name).is_file():
                        self.reply(404, {"error": "Файл не найден."})
                    else:
                        self.send_bytes(200, (report_dir / name).read_bytes(), "text/csv; charset=utf-8", name)
                elif parts.path == "/favicon.ico":
                    self.send_bytes(204, b"", "image/x-icon")
                else:
                    self.reply(404, {"error": "Страница не найдена."})
            except ReviewError as exc:
                self.reply(exc.status, {"error": str(exc), "code": exc.code})
            except (ValueError, KeyError):
                self.reply(400, {"error": "Некорректные параметры запроса."})
            except OSError:
                self.reply(500, {"error": "Файл интерфейса не найден. Перезапустите анализ."})

        def do_POST(self):
            if not self.allowed(post=True):
                return
            if self.path not in ("/api/chat", "/api/reviews"):
                self.reply(404, {"error": "Метод не найден."})
                return
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "Ожидается application/json."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 200_000:
                    self.reply(413, {"error": "Вопрос и история слишком длинные."})
                    return
                body = json.loads(self.rfile.read(size))
                fields = {"message", "history", "context_gid"} if self.path == "/api/chat" else {"gid", "decision", "note"}
                if not isinstance(body, dict) or set(body) - fields:
                    raise ValueError("body")
            except (ValueError, UnicodeError):
                self.reply(400, {"error": "Некорректное тело запроса."})
                return
            if self.path == "/api/reviews":
                try:
                    result = reviews.record(body.get("gid"), body.get("decision"), body.get("note", ""))
                    self.reply(200, {"review": result})
                except ReviewError as exc:
                    self.reply(exc.status, {"error": str(exc), "code": exc.code})
                return
            if not gate.acquire(blocking=False):
                self.reply(429, {"error": "Предыдущий вопрос ещё обрабатывается. Дождитесь ответа.", "code": "busy"})
                return
            try:
                result = factory().chat(body.get("message"), body.get("history"), body.get("context_gid"))
                self.reply(200, result)
            except AssistantError as exc:
                self.reply(exc.status, {"error": str(exc), "code": exc.code})
            except Exception:
                self.reply(500, {"error": "Не удалось обработать ответ модели. Данные не изменены.", "code": "internal"})
            finally:
                gate.release()

    return ThreadingHTTPServer((host, port), Handler)


def serve(data, report, out, port=8520, env_file=".env"):
    server = create_server(data, report, out, port=port, env_file=env_file)
    print(f"След: http://127.0.0.1:{server.server_port}", flush=True)
    print("Для остановки: Ctrl+C. Настройки модели читаются из локального .env.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
