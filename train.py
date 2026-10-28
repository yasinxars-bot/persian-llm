
import os, sys, json, time, signal, threading, subprocess, base64, math, glob
import urllib.request, urllib.error
import torch, torch.nn as nn, torch.nn.functional as F

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GITHUB_TOKEN"]
MODE = os.environ.get("MODE", "train")
PRESET = os.environ.get("PRESET", "persian")
SIG = os.environ.get("SIG", "")
INIT = os.environ.get("INIT", "0")
CUSTOM = os.environ.get("CUSTOM", "0")
PROMPT = os.environ.get("PROMPT", "")
REQ_ID = os.environ.get("REQ_ID", "") or "0"
MAX_SEC = int(os.environ.get("MAX_SEC", "19500"))
SAVE_EVERY = float(os.environ.get("SAVE_EVERY", "3"))
CORPUS_MB = int(os.environ.get("CORPUS_MB", "240"))
CK = "ckpt_work"
API = "https://api.github.com"
stop_flag = threading.Event()
cmd = {"v": "run"}
torch.set_num_threads(os.cpu_count() or 2)


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, method=method, data=data)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "fa-trainer")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = r.read()
            return r.status, (json.loads(d) if d else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def git(*a):
    return subprocess.run(["git", "-C", CK] + list(a), capture_output=True, text=True)


def sync_down():
    os.makedirs(CK, exist_ok=True)
    git("init", "-q")
    git("config", "user.email", "bot@example.com")
    git("config", "user.name", "bot")
    git("remote", "remove", "origin")
    git("remote", "add", "origin", "https://x-access-token:%s@github.com/%s.git" % (TOKEN, REPO))
    r = git("fetch", "-q", "--depth=1", "origin", "ckpt")
    if r.returncode == 0:
        git("checkout", "-q", "-f", "-B", "ckpt", "FETCH_HEAD")
        return True
    return False


push_lock = threading.Lock()
push_ev = threading.Event()
push_count = {"n": 0}


def do_push():
    with push_lock:
        git("add", "-A")
        if git("rev-parse", "HEAD").returncode == 0:
            git("commit", "-q", "--amend", "-m", "ckpt", "--allow-empty")
        else:
            git("commit", "-q", "-m", "ckpt", "--allow-empty")
        r = git("push", "-q", "-f", "origin", "HEAD:ckpt")
        if r.returncode != 0:
            print("push failed:", r.stderr[-300:], flush=True)
        push_count["n"] += 1
        if push_count["n"] % 15 == 0:
            git("reflog", "expire", "--expire=now", "--all")
            git("gc", "-q", "--prune=now")


def pusher():
    while True:
        push_ev.wait()
        push_ev.clear()
        t = time.time()
        do_push()
        time.sleep(max(3.0, 2.0 * (time.time() - t)))


def read_control():
    s, d = api("GET", "/repos/%s/contents/control.json?ref=control" % REPO)
    if s == 200 and "content" in d:
        try:
            return json.loads(base64.b64decode(d["content"])).get("cmd", "run")
        except Exception:
            return None
    return None


def poll_control():
    while True:
        c = read_control()
        if c in ("pause", "stop"):
            cmd["v"] = c
            stop_flag.set()
        time.sleep(5)


class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.l1 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d)
        s.o = nn.Linear(d, d)
        s.l2 = nn.LayerNorm(d)
        s.f1 = nn.Linear(d, 4 * d)
        s.f2 = nn.Linear(4 * d, d)

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.l1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(B, T, C))
        return x + s.f2(F.gelu(s.f1(s.l2(x))))


class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.te = nn.Embedding(256, c["n_embd"])
        s.pe = nn.Embedding(c["block_size"], c["n_embd"])
        s.bl = nn.ModuleList([Block(c["n_embd"], c["n_head"]) for _ in range(c["n_layer"])])
        s.ln = nn.LayerNorm(c["n_embd"])
        s.head = nn.Linear(c["n_embd"], 256, bias=False)

    def forward(s, idx, tg=None):
        T = idx.shape[1]
        x = s.te(idx) + s.pe(torch.arange(T))
        for b in s.bl:
            x = b(x)
        lg = s.head(s.ln(x))
        loss = None if tg is None else F.cross_entropy(lg.reshape(-1, 256), tg.reshape(-1))
        return lg, loss

    @torch.no_grad()
    def generate(s, idx, n, temp=0.8, top_k=40):
        for _ in range(n):
            lg, _ = s(idx[:, -s.c["block_size"]:])
            lg = lg[:, -1] / temp
            v, _ = torch.topk(lg, top_k)
            lg[lg < v[:, [-1]]] = -1e9
            idx = torch.cat([idx, torch.multinomial(F.softmax(lg, -1), 1)], 1)
        return idx


def model_class(use_custom):
    if use_custom and os.path.exists("custom_model.py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_model", "custom_model.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        print("Using custom GPT class from custom_model.py", flush=True)
        return m.GPT
    return GPT


def fetch_wiki():
    out = "corpus_fa.txt"
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        print("pyarrow missing:", e, flush=True)
        return None
    per = CORPUS_MB * 1000000 // 4
    ok = False
    base = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.fa/train-%05d-of-00004.parquet"
    with open(out, "w", encoding="utf-8") as f:
        for k in range(4):
            p = "shard%d.parquet" % k
            for attempt in range(3):
                try:
                    req = urllib.request.Request(base % k, headers={"User-Agent": "fa-trainer"})
                    with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as w:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            w.write(chunk)
                    break
                except Exception as e:
                    print("download retry", k, e, flush=True)
                    time.sleep(5)
            else:
                continue
            got = 0
            try:
                pf = pq.ParquetFile(p)
                for b in pf.iter_batches(batch_size=500, columns=["title", "text"]):
                    for ti, tx in zip(b.column("title").to_pylist(), b.column("text").to_pylist()):
                        s = (ti or "") + "\n" + (tx or "").strip() + "\n\n"
                        f.write(s)
                        got += len(s.encode("utf-8"))
                    if got >= per:
                        break
                ok = ok or got > 0
            except Exception as e:
                print("parquet error", e, flush=True)
            try:
                os.remove(p)
            except Exception:
                pass
    return out if ok else None


def load_data():
    texts = []
    w = fetch_wiki()
    if w:
        texts.append(open(w, encoding="utf-8", errors="ignore").read())
    for f in sorted(glob.glob("data/*.txt")):
        texts.append(open(f, encoding="utf-8", errors="ignore").read())
    raw = "\n".join(texts).encode("utf-8")
    if len(raw) < 1000:
        raise RuntimeError("no corpus available")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8)


def write_atomic(path, data):
    tmp = "tmp_" + os.path.basename(path)
    with open(tmp, "w") as f:
        f.write(data)
    os.replace(tmp, path)


def write_state(cfg, step, loss, epoch, done, preset, params, stage):
    st = {"step": step, "total_steps": cfg["total_steps"], "epoch": round(epoch, 3),
          "loss": round(loss, 4), "done": done, "preset": preset, "sig": SIG,
          "params": params, "stage": stage, "updated": time.time()}
    write_atomic(CK + "/state.json", json.dumps(st))


def save(model, opt, cfg, step, loss, epoch, done, preset, stage):
    params = sum(p.numel() for p in model.parameters())
    d = {"model": model.state_dict(), "cfg": cfg, "step": step, "loss": loss,
         "preset": preset, "sig": SIG, "custom": CUSTOM == "1"}
    if params < 1500000:
        d["opt"] = opt.state_dict()
    torch.save(d, "tmp_model.pt")
    os.replace("tmp_model.pt", CK + "/model.pt")
    write_state(cfg, step, loss, epoch, done, preset, params, stage)


def train():
    c0 = read_control()
    if c0 not in (None, "run"):
        print("control is", c0, "- not starting", flush=True)
        return
    threading.Thread(target=poll_control, daemon=True).start()
    threading.Thread(target=pusher, daemon=True).start()
    resumed = sync_down()
    ck = None
    if resumed and os.path.exists(CK + "/model.pt"):
        c = torch.load(CK + "/model.pt", map_location="cpu", weights_only=False)
        if c.get("sig", "") == SIG:
            ck = c
            print("Resuming from step", ck["step"], flush=True)
        else:
            print("Different model selected - fresh start", flush=True)
    base = json.load(open("models/%s.json" % PRESET))
    init_w = None
    if ck:
        cfg = ck["cfg"]
        preset = ck.get("preset", PRESET)
    else:
        cfg = dict(base)
        preset = PRESET
        if INIT == "1" and os.path.exists("init/model.pt"):
            obj = torch.load("init/model.pt", map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and "cfg" in obj and "model" in obj:
                cfg.update(obj["cfg"])
                init_w = obj["model"]
            else:
                init_w = obj
    model = model_class(CUSTOM == "1")(cfg)
    if init_w is not None:
        try:
            model.load_state_dict(init_w, strict=False)
            print("Loaded initial weights", flush=True)
        except Exception as e:
            print("initial weights ignored:", e, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    step, loss_v, opt_ok = 0, 0.0, False
    if ck:
        model.load_state_dict(ck["model"])
        step, loss_v = ck["step"], ck.get("loss", 0.0)
        if "opt" in ck:
            try:
                opt.load_state_dict(ck["opt"])
                opt_ok = True
            except Exception:
                pass
    n_params = sum(p.numel() for p in model.parameters())
    write_state(cfg, step, loss_v, 0.0, False, preset, n_params, "downloading Persian corpus")
    do_push()
    data = load_data()
    T, B, total = cfg["block_size"], cfg["batch_size"], cfg["total_steps"]
    if len(data) < T + 2:
        data = data.repeat((T + 2) // max(1, len(data)) + 1)
    print("corpus bytes:", len(data), "params:", n_params, flush=True)
    write_state(cfg, step, loss_v, 0.0, False, preset, n_params, "training")
    do_push()
    warm = max(1, min(200, total // 10))
    start_step = step
    rewarm = 0 if (opt_ok or not ck) else 100
    t0 = time.time()
    last = time.time()
    model.train()
    while step < total and not stop_flag.is_set() and time.time() - t0 < MAX_SEC:
        lr = cfg["lr"] * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
        if rewarm:
            lr *= min(1.0, (step - start_step + 1) / rewarm)
        for g in opt.param_groups:
            g["lr"] = lr
        ix = torch.randint(len(data) - T - 1, (B,)).tolist()
        x = torch.stack([data[i:i + T] for i in ix]).long()
        y = torch.stack([data[i + 1:i + T + 1] for i in ix]).long()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        loss_v = loss.item()
        if time.time() - last >= SAVE_EVERY:
            save(model, opt, cfg, step, loss_v, step * B * T / len(data), False, preset, "training")
            push_ev.set()
            last = time.time()
        if step % 50 == 0:
            print("step", step, "loss", round(loss_v, 4), flush=True)
    done = step >= total
    save(model, opt, cfg, step, loss_v, step * B * T / len(data), done, preset,
         "finished" if done else "saved")
    do_push()
    print("exit: step", step, "done", done, "cmd", cmd["v"], flush=True)


def chat():
    if not sync_down() or not os.path.exists(CK + "/model.pt"):
        reply, step = "No checkpoint yet. Press Start and wait for the first save.", 0
    else:
        ck = torch.load(CK + "/model.pt", map_location="cpu", weights_only=False)
        model = model_class(ck.get("custom", False))(ck["cfg"])
        model.load_state_dict(ck["model"])
        model.eval()
        pb = list(PROMPT.encode("utf-8")) or [10]
        idx = torch.tensor([pb], dtype=torch.long)
        out = model.generate(idx, 300)[0].tolist()[len(pb):]
        reply, step = bytes(out).decode("utf-8", "ignore"), ck["step"]
    body = json.dumps({"prompt": PROMPT, "reply": reply, "step": step})
    api("PUT", "/repos/%s/contents/replies/%s.json" % (REPO, REQ_ID),
        {"message": "reply", "branch": "control",
         "content": base64.b64encode(body.encode("utf-8")).decode()})


signal.signal(signal.SIGTERM, lambda *a: stop_flag.set())
signal.signal(signal.SIGINT, lambda *a: stop_flag.set())
if MODE == "chat":
    chat()
else:
    train()
