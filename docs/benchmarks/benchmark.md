# Benchmarking for Speculative Decoding

## Overview

We provide a unified script to test the performance of the Speculative Decoding with EAGLE3 algorithm on multiple datasets. You can follow the steps below to run the benchmarks.

## Run Benchmarks

### Launch SGLang and Benchmarker Concurrently

`bench_eagle3.py` can help you launch a SGLang server process and a Benchmarking process concurrently. In this way, you don't have to launch the SGLang server manually, this script will manually handle the SGLang launch under different speculative decoding configurations. Some important arguments are:
- `--model-path`: the path to the target model.
- `--speculative-draft-model-path`: the path to the draft model.
- `--speculative-bita-model-path`: optional path to the standalone BiTA adapter that should be attached to the draft model.
- For `P-EAGLE`, prepare the checkpoint locally with `scripts/prepare_peagle_checkpoint.py` first.
- `P-EAGLE` currently supports `topk=1` and should be launched with `--disable-overlap-schedule`.
- `--port`: the port to launch the SGLang server.
- `--trust-remote-code`: trust the remote code.
- `--mem-fraction-static`: the memory fraction for the static memory.
- `--tp-size`: the tensor parallelism size.
- `--attention-backend`: the attention backend.
- `--config-list`: the list of speculative decoding configuration to test, the format is `<batch-size>,<num-steps>,<topk>,<num-draft-tokens>`.
- `--benchmark-list`: the list of benchmarks to test, the format is `<benchmark-name>:<num-prompts>:<subset>`.

```shell
python3 bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-bita-model-path /path/to/bita_adapter \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --attention-backend fa3 \
    --config-list 1,0,0,0 1,3,1,4 \
    --benchmark-list mtbench gsm8k:5 ceval:5:accountant \
    --dtype bfloat16
```

### GPT-OSS-20B P-EAGLE

```bash
python scripts/prepare_peagle_checkpoint.py \
    --model-id amazon/GPT-OSS-20B-P-EAGLE \
    --output-dir cache/peagle/GPT-OSS-20B-P-EAGLE

python benchmarks/compare_peagle_vs_eagle3.py \
    --model-path openai/gpt-oss-20b \
    --eagle3-draft-model-path zhuyksir/EAGLE3-gpt-oss-20b-bf16 \
    --peagle-draft-model-path cache/peagle/GPT-OSS-20B-P-EAGLE \
    --eagle3-config-list 1,3,1,4 1,5,1,6 1,7,1,8 \
    --peagle-config-list 1,3,1,4 1,5,1,6 1,7,1,8 \
    --benchmark-list mtbench:80 humaneval:164 \
    --dtype bfloat16
```

### Launch Benchmarker Independently

If you want to launch the SGLang server independently, you can use the following command. Prepend `PYTHONPATH=/path/to/SpecForge`, so the runtime patch can extend the SGLang CLI with the BiTA adapter flag.

```shell
# you can launch a server
PYTHONPATH=/path/to/SpecForge python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-bita-model-path /path/to/bita_adapter \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 1 \
    --tp 1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

Then we can start benchmarking. Note that you should use the same host and port as the one used in the SGLang server. Note that `--skip-launch-server` is required to skip the launch of the SGLang server.

```bash
python bench_eagle3.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --port 30000 \
        --config-list 1,3,1,4 \
        --benchmark-list mtbench:5 ceval:5:accountant gsm8k:5 humaneval:5 math500:5 mtbench:5 aime:1 \
        --skip-launch-server
```
