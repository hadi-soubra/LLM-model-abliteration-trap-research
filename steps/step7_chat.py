"""Step 7: chat with the normal model and the abliterated one, side by side, in the notebook.

    .venv/bin/python steps/step7_chat.py      then open http://localhost:8000/exp1-local.html#chat
    .venv/bin/python steps/step7_chat.py --port 8001     (if port 8000 is taken)

Listens on this computer only. The abliterated version exists only in memory, through the hooks of step 5.
"""
import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import torch

from abliteration.ablation import removing
from shared.config import ROOT, config_from_cli, data_dir
from shared.model import chat_reply, load_model

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
cfg, args = config_from_cli(parser)
port = args.port
prompts = json.loads((data_dir(cfg) / "prompts.json").read_text())
layer = json.loads((data_dir(cfg) / "chosen_layer.json").read_text())["layer"]
r = torch.load(data_dir(cfg) / "directions.pt")[layer]
model, tokenizer = load_model(cfg)
gpu = threading.Lock()  # one answer at a time: there is one model on the GPU


class Handler(SimpleHTTPRequestHandler):
    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/info":  # model name, chosen layer, and example prompts for the page
            examples = {k: prompts[k] for k in ("watch", "harmful_check", "harmless_check")}
            return self.send_json({"model": cfg["model"], "layer": layer, "examples": examples})
        return super().do_GET()  # anything else: a file from report/

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            n = min(int(req.get("max_new_tokens", 256)), 1024)
            with gpu:
                if req["version"] == "abliterated":
                    with removing(model, r):
                        answer = chat_reply(model, tokenizer, req["messages"], n)
                else:
                    answer = chat_reply(model, tokenizer, req["messages"], n)
            self.send_json({"answer": answer})
        except Exception as e:  # show the problem on the page instead of failing silently
            self.send_json({"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):  # keep the terminal quiet
        pass


server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "report")))
print(f"Chat ready: open http://localhost:{port}/exp1-local.html#chat  (layer-{layer} arrow; Ctrl+C to stop)")
server.serve_forever()
