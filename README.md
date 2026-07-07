# CoLabScience: A Proactive AI Assistant for Biomedical Discovery and LLM-Expert Collaborations

[![GitHub](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/YANGWU001/CoLabScience)
[![Paper](https://img.shields.io/badge/Paper-ACL%202026-blue)](https://github.com/YANGWU001/CoLabScience)

**Official implementation of "Excuse me, may I say something... CoLabScience, A Proactive AI Assistant for Biomedical Discovery and LLM-Expert Collaborations"**

*Yang Wu, Jinhong Yu, Jingwei Xiong, Zhimin Tao, Xiaozhong Liu*

**Published in:** Proceedings of the Association for Computational Linguistics (ACL 2026)

---

## Overview

The integration of Large Language Models (LLMs) into scientific workflows presents exciting opportunities to accelerate biomedical discovery. However, the reactive nature of LLMs, which respond only when prompted, limits their effectiveness in collaborative settings that demand foresight and autonomous engagement.

**CoLabScience** is a proactive LLM assistant designed to enhance biomedical collaboration between AI systems and human experts through timely, context-aware interventions. At the core of our method is **PULI** (*Positive-Unlabeled Learning-to-Intervene*), a novel framework trained with a reinforcement learning objective to determine when and how to intervene in streaming scientific discussions, by leveraging the team's project proposal and long- and short-term conversational memory.

### Key Features

- **Proactive Intervention**: Transforms LLMs from reactive tools to active collaborators that autonomously identify intervention opportunities
- **PULI Framework**: Positive-Unlabeled Learning-to-Intervene with reinforcement learning for when-and-how intervention decisions
- **Dual-LLM Architecture**: Efficient **Observer** (LLaMA-3.2-1B) for intervention timing + powerful **Presenter** (LLaMA-3.1-8B) for content generation
- **BSDD Benchmark**: *Biomedical Streaming Dialogue Dataset* of simulated multi-role research dialogues grounded in PubMed literature
- **End-to-End Training**: Coordinator, Observer, and Presenter jointly optimized via SFT, GRPO, and REINFORCE
- **Dual-Scale Memory**: Long-term and short-term conversational memory for context-aware interventions

This repository provides all necessary scripts to reproduce the training pipeline and experimental results from our ACL 2026 paper.

---

## Environment Setup

Install the required dependencies using the provided Conda environment file:

```bash
conda env create -f environment.yml
conda activate colabscience
```

Install **LLaMA Factory** (used for Presenter SFT training):

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
```

### Requirements

- Python 3.10+
- PyTorch 2.6.0
- Transformers 4.51.3
- CUDA-capable GPU(s) — **8× A100 80GB recommended**
- See `environment.yml` for complete dependencies

---

## Dataset

We introduce **BSDD** (*Biomedical Streaming Dialogue Dataset*), a benchmark of simulated biomedical research dialogues with proactive intervention labels derived from PubMed articles.

**Note**: This repository contains a high-quality subset of the full BSDD dataset. Due to the large size of the complete dataset, we release a randomly sampled subset that maintains the original distribution and quality.

| File | Description |
|------|-------------|
| `data/train.json` | Training data in PU learning format (labeled positive + unlabeled samples) |
| `data/validation.json` | Balanced evaluation data (positive/negative) |
| `data/test.json` | Balanced evaluation data (positive/negative) |

Each sample includes project context, long-term memory, recent conversation history, and intervention labels.

---

## Quick Start

### Start End-to-End Training

```bash
cd code
python start_training.py
```

The training process automatically handles:
- PU learning on unlabeled data via the Coordinator
- SFT of the Presenter using potential positives
- GRPO training of the Observer for intervention judgment
- REINFORCE optimization of the Coordinator using validation performance

---

## Training Pipeline

The CoLabScience framework consists of four core components and an end-to-end training loop:

### Architecture

| Component | Model | Role |
|-----------|-------|------|
| **Observer** | LLaMA-3.2-1B-Instruct | Determines *when* to intervene |
| **Presenter** | LLaMA-3.1-8B-Instruct | Generates *how* to intervene |
| **Linear Projector** | Learnable projection layer | Aligns embeddings between Observer and Presenter |
| **Coordinator** | 6-layer MLP | Distinguishes potential positives from negatives in PU data |

### Step 1: Coordinator Training

Process unlabeled data through Observer + Presenter → embeddings → Coordinator → action probabilities. Samples with action ≥ 0.5 are treated as potential positives; action < 0.5 as potential negatives.

### Step 2: SFT Phase

Potential positives augment labeled positive data for Presenter fine-tuning via LLaMA Factory:

```bash
cd code/llamafactory_configs
./quick_train_presenter.sh
```

**Output**: Checkpoints saved in `./code/saved_models/presenter_sft/`

### Step 3: GRPO Phase

Potential negatives form the PN (Positive-Negative) dataset for Observer intervention judgment training via GRPO (Group Relative Policy Optimization).

**Output**: Observer checkpoints saved in `./code/saved_models/observer/`

### Step 4: REINFORCE Phase

Observer and Presenter validation performance provides a combined reward signal (weighted by λ=0.6) to optimize the Coordinator in an end-to-end loop.

---

## GPU Requirements

Recommended configuration (8× A100 80GB):

| Phase | GPU Allocation |
|-------|----------------|
| **SFT** | GPUs 0–1 (LLaMA Factory training) |
| **GRPO** | GPU 3 (vLLM server), GPUs 4–7 (model training) |
| **Main Training** | Configurable via `CUDA_VISIBLE_DEVICES` in `start_training.py` |

---

## Project Structure

```
CoLabScience/
├── data/                                    # BSDD dataset files
│   ├── train.json                           # PU-format training data
│   ├── validation.json                      # Balanced validation set
│   └── test.json                            # Balanced test set
├── code/                                    # Core implementation
│   ├── config.py                            # Configuration parameters
│   ├── models.py                            # Observer, Presenter, Coordinator, Projector
│   ├── train.py                             # Main end-to-end training loop
│   ├── start_training.py                    # Training launcher
│   ├── grpo_auto.py                         # Automated GRPO training
│   ├── grpo_trainer.py                      # GRPO training implementation
│   ├── accelerate_configs/                  # DeepSpeed configurations
│   │   ├── deepspeed_zero3.json
│   │   └── deepspeed_zero3.yaml
│   └── llamafactory_configs/                # LLaMA Factory SFT configs
│       ├── presenter_sft_config.yaml
│       ├── dataset_info.json
│       └── quick_train_presenter.sh
├── environment.yml                          # Conda environment configuration
└── README.md
```

---

## Configuration

Key hyperparameters can be adjusted in `code/config.py`:

```python
# Action threshold for positive/negative classification
action_threshold = 0.5

# Reward weighting between Observer and Presenter
reward_weight_lambda = 0.6

# Training epochs
total_training_epochs = 30

# Base models
observer_model_name = "meta-llama/Llama-3.2-1B-Instruct"
presenter_model_name = "meta-llama/Llama-3.1-8B-Instruct"
```

---

## Results

CoLabScience demonstrates significant improvements in proactive biomedical collaboration:

- **Intervention Precision**: PULI significantly outperforms existing baselines in intervention timing accuracy
- **Collaborative Task Utility**: Strong performance on both simulation-based and human evaluation
- **Generalizability**: Robust across a range of LLM backbones
- **Efficiency**: Observer-Presenter architecture enables real-time monitoring with on-demand content generation

See the paper for detailed experimental results and analysis.

---

## Notes

- This implementation uses LLaMA-3.2-1B as the Observer and LLaMA-3.1-8B as the Presenter by default.
- The BSDD dataset is released alongside this codebase for proactive scientific assistant research.
- Human evaluation details and prompt templates are provided in the paper appendix.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wu2026colabscience,
  title={"Excuse me, may I say something..." CoLabScience, A Proactive {AI} Assistant for Biomedical Discovery and {LLM}-Expert Collaborations},
  author={Wu, Yang and Yu, Jinhong and Xiong, Jingwei and Tao, Zhimin and Liu, Xiaozhong},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL 2026)},
  year={2026}
}
```

---

## BSDD Dataset

The **BSDD (Biomedical Streaming Dialogue Dataset)** benchmark is designed for training and evaluating proactive LLM assistants in biomedical research collaboration. It includes:

- Multi-role simulated research dialogues (Pharmacologist, Medicinal Chemist, Bioinformatician, Clinical Physician)
- Project proposals grounded in PubMed literature
- Positive-unlabeled intervention labels with sparse annotation strategy
- Long-term and short-term conversational memory annotations

The dataset is available in the `data/` directory.

---

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors via the paper correspondence

**GitHub**: [https://github.com/YANGWU001/CoLabScience](https://github.com/YANGWU001/CoLabScience)

---

## License

This project is released for research purposes. Please cite our paper if you use this code or the BSDD dataset in your research.

---
