"""Restart vLLM across the four serving configurations and run Study 4.

This driver keeps the traffic workload identical while changing one server
knob at a time. It owns only the vLLM processes it starts and terminates their
process group after each configuration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


CONFIGS = {
    "baseline": {"max_num_seqs": 64, "gpu_mem_util": 0.9, "chunked": True},
    "seq16": {"max_num_seqs": 16, "gpu_mem_util": 0.9, "chunked": True},
    "memory50": {"max_num_seqs": 64, "gpu_mem_util": 0.5, "chunked": True},
    "no_chunk": {"max_num_seqs": 64, "gpu_mem_util": 0.9, "chunked": False},
}


def server_healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_for_server(process: subprocess.Popen, base_url: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {process.returncode}")
        if server_healthy(base_url):
            return
        time.sleep(1)
    raise RuntimeError(f"vLLM did not become healthy within {timeout_s} seconds")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--vllm", default="/workspace/vllm-env/bin/vllm")
    parser.add_argument("--client-python", default="/workspace/vllm-env/bin/python")
    parser.add_argument("--plot-python", default="/venv/main/bin/python")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4608)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--patterns", default="poisson,bursty")
    parser.add_argument("--rates", default="0.5,1,2,3,4,6")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--hf-home", default="/workspace/hf-cache")
    parser.add_argument("--server-timeout", type=int, default=300)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing load CSVs when the first requested config runs",
    )
    args = parser.parse_args()

    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        parser.error(f"unknown configs: {sorted(unknown)}")
    base_url = f"http://{args.host}:{args.port}"
    env = os.environ.copy()
    env["HF_HOME"] = args.hf_home
    Path("results").mkdir(exist_ok=True)

    for index, name in enumerate(configs):
        config = CONFIGS[name]
        log_path = Path("results") / f"load_server_{name}.log"
        server_command = [
            args.vllm,
            "serve",
            args.model,
            "--served-model-name",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--dtype",
            "float16",
            "--gpu-memory-utilization",
            str(config["gpu_mem_util"]),
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(config["max_num_seqs"]),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--no-enable-prefix-caching",
            (
                "--enable-chunked-prefill"
                if config["chunked"]
                else "--no-enable-chunked-prefill"
            ),
        ]
        print(f"starting {name}: {' '.join(server_command)}", flush=True)
        with log_path.open("w") as log_handle:
            process = subprocess.Popen(
                server_command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        try:
            wait_for_server(process, base_url, args.server_timeout)
            client_command = [
                args.client_python,
                "scripts/bench_load.py",
                "--base-url",
                base_url,
                "--model",
                args.model,
                "--server-config",
                name,
                "--max-num-seqs",
                str(config["max_num_seqs"]),
                "--gpu-mem-util",
                str(config["gpu_mem_util"]),
                (
                    "--chunked-prefill"
                    if config["chunked"]
                    else "--no-chunked-prefill"
                ),
                "--patterns",
                args.patterns,
                "--rates",
                args.rates,
                "--requests",
                str(args.requests),
                "--repeats",
                str(args.repeats),
            ]
            if args.overwrite and index == 0:
                client_command.append("--overwrite")
            subprocess.run(client_command, check=True, env=env)
        except Exception:
            print(f"server log tail ({log_path}):\n{tail(log_path)}", flush=True)
            raise
        finally:
            stop_server(process)
        print(f"finished {name}", flush=True)
        time.sleep(5)

    subprocess.run(
        [args.plot_python, "scripts/plot_load.py"],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
