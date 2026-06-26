"""多线程分块下载 torch/torchvision cu121 wheel。

加固版：网络反复断连(IncompleteRead/ProtocolError)时，每块独立重试+断点续传，
pip 扛不住的断连，这里能扛。零依赖（仅 Python 标准库）。
"""
import os
import time
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

THREADS = 16
RETRIES = 20
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")

FILES = [
    ("https://download.pytorch.org/whl/cu121/torch-2.3.1%2Bcu121-cp310-cp310-win_amd64.whl",
     "torch-2.3.1+cu121-cp310-cp310-win_amd64.whl"),
    ("https://download.pytorch.org/whl/cu121/torchvision-0.18.1%2Bcu121-cp310-cp310-win_amd64.whl",
     "torchvision-0.18.1+cu121-cp310-cp310-win_amd64.whl"),
]


def head_size(url):
    for _ in range(RETRIES):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except Exception:
            time.sleep(2)
    raise RuntimeError("HEAD failed: " + url)


def fetch_range(url, idx, start, end, part_path):
    want = end - start + 1
    for attempt in range(RETRIES):
        try:
            existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if existing >= want:
                return idx, part_path
            cur_start = start + existing
            req = urllib.request.Request(url, headers={"Range": f"bytes={cur_start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as r, open(part_path, "ab") as f:
                while True:
                    chunk = r.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(part_path) >= want:
                return idx, part_path
        except Exception as e:
            print(f"  part {idx} attempt {attempt+1} 断连续传: {type(e).__name__}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"range {idx} failed after {RETRIES} retries: {url}")


def download(url, filename, n=THREADS):
    dest = os.path.join(DEST, filename)
    if os.path.exists(dest):
        print(f"[skip] {filename} exists", flush=True)
        return dest
    size = head_size(url)
    print(f"[start] {filename} {size / 1e6:.1f}MB x{n} threads", flush=True)
    chunk = math.ceil(size / n)
    tasks = []
    for i in range(n):
        s = i * chunk
        e = min((i + 1) * chunk - 1, size - 1)
        if s > e:
            break
        tasks.append((i, s, e))
    parts = {}

    def work(t):
        i, s, e = t
        p = os.path.join(DEST, f"{filename}.part{i}")
        return fetch_range(url, i, s, e, p)

    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            idx, p = fut.result()
            parts[idx] = p
            print(f"  part {idx} done", flush=True)
    with open(dest, "wb") as out:
        for i in sorted(parts):
            with open(parts[i], "rb") as f:
                while True:
                    c = f.read(16 * 1024 * 1024)
                    if not c:
                        break
                    out.write(c)
            os.remove(parts[i])
    print(f"[done] {filename} {os.path.getsize(dest) / 1e6:.1f}MB", flush=True)
    return dest


os.makedirs(DEST, exist_ok=True)
for url, fn in FILES:
    download(url, fn)
print("ALL DONE", flush=True)
