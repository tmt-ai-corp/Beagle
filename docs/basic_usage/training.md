## 🚀 Training

## 📍 Overview

Existing speculative decoding methods such as EAGLE3 requires training in the feature-space, which means the draft model relies on the hidden states generated from the target model for autoregressive prediction. In SpecForge, we provide two orthogonal paths to cater to the users' specific needs when training this kind of draft models. We name these two methods as `Online` and `Offline`. By definition, it is easy to understandd them:

- **`Online`**: the hidden states are generated on the fly during training.
- **`Offline`**: the hidden states are generated beforehand, stored to the disk, and loaded back to GPU during training.

Online training is suitable for users with limited disk space but sufficient GPUs while offline training is suitable for users with sufficient disk space but limited GPUs.

| Method | Target Model | Disk Space Requirement | GPU Requirement | One-liner rationale |
| --- | --- | --- | --- | --- |
| Online | Used during training | Small | More GPUs are needed if your target model is large | Generating auxiliary hidden states on the fly |
| Offline | Only used during data preparation | Huge (e.g. ultrachat+sharegpt will need 12TB storage ) | as low as 1 GPU, as only need to accommodate the draft model  | Preparing auxiliary hidden states beforehand and only once |

> **Why does disk matter?**
> During Eagle3 training, the frozen target model will first generate the hidden states for each token given the data sample. The hidden states are fed to the draft model for training.
> Offline mode stores these hidden states to the local disk, so a small disk can be filled up fast.
> Online mode only generates these hidden states on the fly without storing them to the disk, but needs to keep the target model resident in memory during training, trading GPU RAM for almost-zero disk footprint.

## 🏎️ Online Training

We have provided training scripts for the EAGLE3 models in the `examples` directory. These scripts cover a wide range of models range from Llama to Qwen, small to large and dense to MoE. Online training is often conducted in two steps and we will use ShareGPT and Llama3-8B-Instruct as an example.

**Step 1: Prepare the dataset**

```bash
# prepare the dataset
python scripts/prepare_data.py --dataset sharegpt
```

**Step 2: Start the training**

```bash
# train llama3-8B-instruct
bash ./examples/run_llama3.1_8b_eagle3_online.sh
```

### BiTA-style auxiliary training on top of an existing Eagle3 draft

For research experiments where you want to continue from an already-trained Eagle3 drafter and add a BiTA-style prompt/mask auxiliary objective, use the H100-oriented example script:

```bash
bash ./examples/run_llama3.1_8b_eagle3_bita_online_h100.sh
```

This path currently has a few intentional constraints:

- It is only enabled for the `sdpa` draft attention backend.
- It is currently wired for text-only draft training, not VLM training.
- The script expects a frozen base Eagle3 draft checkpoint in `BASE_DRAFT_CKPT`.
- The training output directory stores the standalone BiTA adapter checkpoint, not a rewritten full draft checkpoint.
- You can later reconstruct the full experiment by loading the frozen base draft from `BASE_DRAFT_CKPT` and the trained BiTA adapter from the BiTA output directory.

### OneBit linear conversion on top of an existing Eagle3 draft

For a second-stage experiment where you start from a trained dense Eagle3 draft, convert its linear layers to OneBit modules, and continue with the standard Eagle3 loss, use:

```bash
bash ./examples/run_llama3.1_8b_eagle3_onebit_online_h100.sh
```

This path is intentionally independent from the BiTA path:

- It expects a trained dense draft checkpoint in `BASE_DRAFT_CKPT`.
- The script converts the draft's linear layers in-place to OneBit modules and then runs the normal Eagle3 objective.
- The saved checkpoints are full draft checkpoints with `use_onebit=true` recorded in `config.json`, so they can be resumed directly with `from_pretrained`.
- By default, `lm_head` stays dense for stability. Set `ONEBIT_INCLUDE_LM_HEAD=1` if you want to quantize it too.

## 💨 Offline Training

The difference between online and offline training is that we need to generate the hidden states before training. We also use ShareGPT and Llama3-8B-Instruct as an example.

**Step 1: Prepare the dataset**

Same as above

**Step 2: Generate the hidden states and train**

```bash
# train llama3-8B-instruct in an offline manner
bash ./examples/run_llama3.1_8b_eagle3_offline.sh
```

It is important to note that the `run_llama3.1_8b_eagle3_offline.sh` script consists of two steps:

1. Generate the hidden states using the `prepare_hidden_states.py` script. This script will generate the hidden states for the test and train datasets and save them to the disk.
2. Train the model: suppling the `--train-hidden-states-path` argument to the script so that the script will load the hidden states from the disk during training.

## 📈 Experiment Tracking

This project supports logging training progress to Wandb, TensorBoard, and SwanLab. You can enable tracking by adding the `--report-to` argument to the command line in your shell script.
