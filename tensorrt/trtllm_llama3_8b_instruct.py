import os
import time

import modal
from pathlib import Path


ENGINE_DIR = "/root/model/model_output"
BASE_MODEL = "agyaatcoder/llama3-8b-instruct-A100-trtllm"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

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

stub = modal.Stub(
    f"example-trtllm-{MODEL_ID}", image= tensorrt_image
)

GPU_CONFIG = modal.gpu.A100(count=1) 

@stub.cls(gpu=GPU_CONFIG, secrets=[modal.Secret.from_name("huggingface-llama")])
class Model:
    @modal.enter()
    def load(self):
        import time
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.pad_id = self.tokenizer.eos_token_id
        self.end_id = self.tokenizer.eos_token_id
        
        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}/compiled-model",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

    @modal.method()
    def generate(self, prompts):
        settings = dict(
            temperature=0.9,
            top_k=1, 
            top_p=0,
            max_new_tokens=256,
            end_id = self.end_id,
            pad_id = self.pad_id,
            stop_words_list = None
        )
        
        start = time.monotonic_ns()

        results = []
        for prompt in prompts:
            
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )


            output_ids = self.model.generate(
                input_ids,
                **settings,
            )
            #print(f"output: {output}\n")
            #print(output[0][0])
            #TODOS, use input tensor size for output postprocessing
            #batch_size, num_beams, _ = output_ids.size()
            #output_begin = input_lengths[batch_idx]
        
            output_text = self.tokenizer.decode(output_ids[0][0])


            #TODOS Use stop_sequence - tensor for post processing
           
            assistant_response = self.extract_assistant_response(output_text)
            results.append(assistant_response)
            #results.append(output_text)

        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = len(output_ids[0][0]) # contains input tokens too 
        num_tokens_processed = sum(len(r) for r in results)
        
        print(f"Generated {num_tokens} tokens in {duration_s:.1f} seconds, "
              f"throughput = {num_tokens/duration_s:.0f} tokens/second on {GPU_CONFIG}.",
              f"Generated processed {num_tokens_processed} tokens in {duration_s:.1f} seconds, "
              )
        
        return results
    
    
    def extract_assistant_response(self, output_text):
        # Split the output text by the assistant header token
        parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")
        
        if len(parts) > 1:
            # Join the parts after the first occurrence of the assistant header token
            response = parts[1].split("<|eot_id|>")[0].strip()
            
            # Remove any remaining special tokens and whitespace
            response = response.replace("<|eot_id|>", "").strip()
            
            return response
        else:
            return output_text    

@stub.local_entrypoint()
def main():
    questions = [
        # Coding questions
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What are the differences between Javascript and Python?",
        "How do I find invalid indices in Postgres?",
        "How can you implement a LRU (Least Recently Used) cache in Python?",
        "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        "Can you explain how a decision tree algorithm works in machine learning?",
        "How would you design a simple key-value store database from scratch?",
        "How do you handle deadlock situations in concurrent programming?",
        "What is the logic behind the A* search algorithm, and where is it used?",
        "How can you design an efficient autocomplete system?",
        "What approach would you take to design a secure session management system in a web application?",
        "How would you handle collision in a hash table?",
        "How can you implement a load balancer for a distributed system?",
        # Literature
        "What is the fable involving a fox and grapes?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        "Who does Harry turn into a balloon?",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        "Describe a day in the life of a secret agent who's also a full-time parent.",
        "Create a story about a detective who can communicate with animals.",
        "What is the most unusual thing about living in a city floating in the clouds?",
        "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        "Tell a story about a musician who discovers that their music has magical powers.",
        "In a world where people age backwards, describe the life of a 5-year-old man.",
        "Create a tale about a painter whose artwork comes to life every night.",
        "What happens when a poet's verses start to predict future events?",
        "Imagine a world where books can talk. How does a librarian handle them?",
        "Tell a story about an astronaut who discovered a planet populated by plants.",
        "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        "Write a tale about a chef whose food can evoke memories from the eater's past.",
        # History
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        "How did the Greek philosophers influence economic thought in the ancient world?",
        "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        "How did decolonization in the 20th century change the geopolitical map?",
        "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        "What could be the potential implications of contact with an advanced alien civilization?",
        "Describe how you would mediate a conflict between two roommates about doing the dishes using techniques of non-violent communication.",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # Facts
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        "What was Project A119 and what were its objectives?",
        "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        "What is the 'Emu War' that took place in Australia in the 1930s?",
        "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
        # Multilingual
        "战国时期最重要的人物是谁?",
        "Tuende hatua kwa hatua. Hesabu jumla ya mfululizo wa kihesabu wenye neno la kwanza 2, neno la mwisho 42, na jumla ya maneno 21.",
        "Kannst du die wichtigsten Eigenschaften und Funktionen des NMDA-Rezeptors beschreiben?",
    ]
    model = Model()
    results = model.generate.remote(questions)
    for prompt, result in zip(questions, results):
        print(f"Prompt: {prompt}\nCompletion: {result}\n")




