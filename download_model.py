"""可断点续传的多线程下载 SD1.5 模型。

解决两个问题:
1. 迅雷被 hf-mirror 封 UA (报"原始资源不存在") -> 用普通 HTTP 请求 + 浏览器 UA
2. CDN 断连 (IncompleteRead/ProtocolError) -> 每块独立重试 + 断点续传

零依赖，仅 Python 标准库。用 venv 跑: ./venv/Scripts/python.exe download_model.py
"""
import os
import time
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
DEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "Stable-diffusion", "v1-5-pruned-emaonly.safetensors",
)
EXPECT = 4265146304  # 官方 v1-5-pruned-emaonly 字节数
THREADS = 8
RETRIES = 40
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def head_size() -> int:
    for _ in range(RETRIES):
        try:
            req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except Exception:
            time.sleep(2)
    raise RuntimeError("HEAD failed after retries")


def fetch_range(i: int, start: int, end: int, part_path: str):
    want = end - start + 1
    for attempt in range(RETRIES):
        try:
            existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            if existing >= want:
                return i, part_path
            cur_start = start + existing
            req = urllib.request.Request(
                URL, headers={"Range": f"bytes={cur_start}-{end}", "User-Agent": UA}
            )
            with urllib.request.urlopen(req, timeout=300) as r, open(part_path, "ab") as f:
                while True:
                    chunk = r.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(part_path) >= want:
                return i, part_path
        except Exception as e:
            print(f"  part {i} attempt {attempt+1} 续传: {type(e).__name__}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"part {i} failed after {RETRIES} retries")


def main() -> None:
    if os.path.exists(DEST) and os.path.getsize(DEST) == EXPECT:
        print(f"[skip] 已完整: {DEST}", flush=True)
        return
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    size = head_size()
    print(f"[start] {size/1e9:.2f}GB x{THREADS} threads -> {os.path.basename(DEST)}", flush=True)
    if size != EXPECT:
        print(f"[warn] HEAD 报告大小 {size} 与预期 {EXPECT} 不符，按实际下载", flush=True)

    chunk = math.ceil(size / THREADS)
    tasks = []
    for i in range(THREADS):
        s = i * chunk
        e = min((i + 1) * chunk - 1, size - 1)
        if s > e:
            break
        tasks.append((i, s, e))

    parts: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = {ex.submit(fetch_range, i, s, e, f"{DEST}.part{i}"): i for i, s, e in tasks}
        for fut in as_completed(futs):
            i, p = fut.result()
            parts[i] = p
            done = sum(os.path.getsize(parts[k]) for k in parts)
            print(f"  part {i} done  ({done/1e9:.2f}/{size/1e9:.2f}GB)", flush=True)

    with open(DEST, "wb") as out:
        for i in sorted(parts):
            with open(parts[i], "rb") as f:
                while True:
                    c = f.read(16 * 1024 * 1024)
                    if not c:
                        break
                    out.write(c)
            os.remove(parts[i])

    got = os.path.getsize(DEST)
    print(f"[done] {got} / {EXPECT} bytes  {'OK' if got == EXPECT else 'SIZE MISMATCH'}", flush=True)


main()
