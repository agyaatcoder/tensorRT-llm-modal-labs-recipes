

import os
import subprocess
import time

import modal

ENGINE_DIR = "/root/model/model_output"
BASE_MODEL = "agyaatcoder/llama3-8b-instruct-A100-trtllm"

def download_compiled_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(ENGINE_DIR, exist_ok=True)
    snapshot_download(
        BASE_MODEL,
        local_dir=ENGINE_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()

tensorrt_image = (
    modal.Image.from_registry( "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("openmpi-bin", "libopenmpi-dev")
    .apt_install("git", "git-lfs", "wget")
    .apt_install("python3-pip")
    .run_commands("pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com")
    .pip_install(
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "requests",
        "fastapi")
    .pip_install("jupyterlab")
    .run_function(download_compiled_model_to_folder, timeout=60 * 20))


        
GPU_CONFIG = modal.gpu.A100() #40GB memory

JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!

stub = modal.Stub("tensorrt-jupy", image =tensorrt_image )



@stub.function(concurrency_limit=1, timeout=3_000, gpu = GPU_CONFIG, secrets=[modal.Secret.from_name("huggingface-llama"), modal.Secret.from_name("huggingface-write")])
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@stub.local_entrypoint()
def main(timeout: int = 10_000):
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)
