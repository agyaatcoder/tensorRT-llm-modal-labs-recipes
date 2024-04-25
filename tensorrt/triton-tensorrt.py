# Triton inference server from TensorRT-LLM engine
import os
import subprocess
from modal import Image, Secret, Stub, gpu, web_server
from pathlib import Path
from itertools import islice



ENGINE_DIR = "/root/model_output"
BASE_MODEL = "agyaatcoder/llama3-8b-instruct-A100-trtllm"

# Define configurable variables
TOKENIZER_DIR = "meta-llama/Meta-Llama-3-8B-Instruct"
TOKENIZER_TYPE = "llama"

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

def download_tensorrtllm_backend():
    import subprocess

    # Clone the TensorRT-LLM backend repository
    repo_url = "https://github.com/triton-inference-server/tensorrtllm_backend.git"
    branch = "release/0.5.0"
    clone_command = f"git clone -b {branch} {repo_url}"
    try:
        subprocess.run(clone_command, shell=True, check=True)
        print(f"Repository {repo_url} cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr}")

    # Copy the compiled model assets
    copy_command = "cp /root/model_output/compiled-model/* /root/tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/"
    try:
        subprocess.run(copy_command, shell=True, check=True)
        print("Compiled model assets copied successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error copying model assets: {e.stderr}")

    # Modify configuration files
    config_commands = [
        "python3 tools/fill_template.py --in_place all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt decoupled_mode:true,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1,max_tokens_in_paged_kv_cache:,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.2,max_num_sequences:4",
        f"python tools/fill_template.py --in_place all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_type:{TOKENIZER_TYPE},tokenizer_dir:{TOKENIZER_DIR}",
        f"python tools/fill_template.py --in_place all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_type:{TOKENIZER_TYPE},tokenizer_dir:{TOKENIZER_DIR}"
    ]

    for command in config_commands:
        try:
            subprocess.run(f"cd tensorrtllm_backend && {command}", shell=True, check=True)
            print(f"Configuration command executed successfully: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing configuration command: {e.stderr}")

triton_image = (
    Image.from_registry(
        "nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3", add_python="3.10"
    )
    #.apt_install("openmpi-bin", "libopenmpi-dev")
    .apt_install("git")
    .pip_install(
        "sentencepiece",
        "protobuf",
        "transformers==4.39.3",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_compiled_model_to_folder,
        secrets=[Secret.from_name("hf-secret-llama")],
        timeout=60 * 20,
    )
    .run_function(download_tensorrtllm_backend)

)

stub = Stub("Triton-tensorRT", image=triton_image)
GPU_CONFIG = gpu.A100()  # 40GB VRAM

@stub.function(
    allow_concurrent_inputs=10,
    gpu=GPU_CONFIG,
    cpu= 1
)
@web_server(8000, startup_timeout=60)
def triton_server():
    #cmd = f"python /root/tensorrtllm_backend/scripts/launch_triton_server.py --model_repo /root/tensorrtllm_backend/all_models/inflight_batcher_llm --world_size 1"
    import os

    #os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = str(1)
    #os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = str(1)
   

    
    #cmd = f"mpirun --allow-run-as-root -n 1 --oversubscribe /opt/tritonserver/bin/tritonserver --model_repository=/root/tensorrtllm_backend/all_models/inflight_batcher_llm  --model-control-mode=explicit --load-model=preprocessing --load-model=postprocessing --load-model=tensorrt_llm --load-model=tensorrt_llm_bls --load-model=ensemble  --log-verbose=2 --log-info=1 --log-warning=1 --log-error=1"
    cmd = f"mpirun --allow-run-as-root -n 1 --oversubscribe /opt/tritonserver/bin/tritonserver --model-store /root/tensorrtllm_backend/all_models/inflight_batcher_llm  --model-control-mode explicit --load-model preprocessing --load-model postprocessing --load-model=tensorrt_llm --load-model tensorrt_llm_bls --load-model ensemble  --log-verbose 2 --log-info 1 --log-warning 1 --log-error 1"
    subprocess.Popen(cmd, shell=True)
    #tritonserver --model-repository=/tensorrt/triton-repos/trtibf-Trendyol-LLM-7b-chat-v1.0 --model-control-mode=explicit --load-model=preprocessing --load-model=postprocessing --load-model=tensorrt_llm --load-model=tensorrt_llm_bls --load-model=ensemble  --log-verbose=2 --log-info=1 --log-warning=1 --log-error=1
