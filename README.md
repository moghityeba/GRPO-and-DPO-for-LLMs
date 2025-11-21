# Improving LLMs with RLHF: DPO & GRPO

## Introduction

This project demonstrates two post-training techniques for improving Large Language Models through Reinforcement Learning from Human Feedback (RLHF):

- **Part I: Direct Preference Optimization (DPO)** — Fine-tuning Qwen2.5-0.5B-Instruct on French preference data
- **Part II: Group Relative Policy Optimization (GRPO)** — Training on GSM8K math reasoning tasks

Both methods align language models with human preferences while maintaining efficiency through parameter-efficient fine-tuning techniques.

## Part I: DPO (Direct Preference Optimization)

### Overview

Fine-tune **Qwen2.5-0.5B-Instruct** on French-language data using off-policy DPO to improve instruction-following and language-specific understanding.

### Key Concepts

- **Model:** Qwen2.5-0.5B-Instruct
- **Objective:** Adapt model for French understanding and instruction-following
- **Method:** Off-policy DPO for alignment-based fine-tuning
- **Optimization:** LoRA (Low-Rank Adaptation) + 4-bit quantization (QLoRA)

### Methodology

#### 1. Data Preparation

Modern instruction-tuned models expect inputs in chat format. We structure each example as:
```python
messages = [
  {"role": "system", "content": "Tu es un assistant utile. Réponds en français."},
  {"role": "user", "content": "Explique la différence entre LoRA et le fine-tuning complet."},
  {"role": "assistant", "content": "LoRA adapte un petit sous-espace de poids..."}
]
```

**Dataset Requirements:**
- `prompt`: shared context (system + user turns)
- `chosen`: preferred assistant reply
- `rejected`: less-preferred reply

#### 2. Training Setup

We use the `trl` library with:
- **Policy model:** Trainable Qwen2.5-0.5B-Instruct with LoRA adapters
- **Reference model:** Frozen copy for KL divergence constraint
- **Loss function:** DPO objective optimizing preference pairs

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

Where:
- $y_w$ = chosen response, $y_l$ = rejected response
- $\beta$ = KL penalty coefficient
- $\pi_\theta$ = policy model, $\pi_{\text{ref}}$ = reference model

#### 3. Memory Optimization

- **4-bit quantization** to reduce VRAM usage
- **LoRA adapters** for parameter-efficient training
- **W&B logging** for metrics and artifacts

## Part II: GRPO (Group Relative Policy Optimization)

### Overview

GRPO is a reinforcement learning approach that optimizes generation quality directly from preference data using policy gradients and reward modeling.

**Key Differences from DPO:**
- On-policy updates (samples from current model)
- Combines PPO-style optimization with preference rewards
- Better captures dynamic generation quality aspects

### Dataset: GSM8K

Grade-school math word problems where the model must:
1. Reason through the problem (in French or English)
2. Output the final numeric answer in format: `<answer>NUMBER</answer>`

### Reward Function Design

We use two complementary rewards during rollouts:

**1. Format Reward** — Checks correct output structure

$$
r_{\text{format}} = \begin{cases} 
1 & \text{if last line is } \texttt{<answer>NUMBER</answer>} \\ 
0 & \text{otherwise} 
\end{cases}
$$

**2. Correctness Reward** — Validates answer accuracy

$$
r_{\text{correct}} = \begin{cases} 
2 & \text{if extracted number matches gold answer} \\ 
0 & \text{otherwise} 
\end{cases}
$$

**Total reward per sample:**

$$
r_{\text{total}} = r_{\text{format}} + r_{\text{correct}} \in \{0, 1, 2, 3\}
$$

### Training Objectives

The model learns to:
- Follow instructions to output in the correct format
- Develop mathematical reasoning capabilities
- Generate structured answers for automatic evaluation

## System Requirements

### Hardware
- **Recommended:** GPU with 16GB+ VRAM (tested on RunPod)
- **Memory optimization:** QLoRA enables training on consumer GPUs

