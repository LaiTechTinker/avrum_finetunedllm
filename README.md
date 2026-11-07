
# Avrum Agricultural LLM Training Pipeline

A reusable   pipeline for fine-tuning an open-source language model on agricultural question-answering data.

This project converts the original Jupyter notebook training experiment into a modular Python pipeline.

The goal is to produce an agriculture-focused QLoRA adapter that can eventually serve as a fallback language model for Avrum AI.

---

## 1. Project Overview

### Base Model
- Qwen/Qwen2-1.5B-Instruct

### Dataset
- KisanVaani/agriculture-qa-english-only

### Fine-tuning Method
- QLoRA
- 4-bit NF4 quantization
- LoRA adapters

### Dataset Preparation
- 8,000 selected samples
- Random seed: 42
- 90% training split
- 10% validation split

### Training
- Supervised Fine-Tuning (SFT)
- Hugging Face Transformers
- Hugging Face TRL
- PEFT

All training parameters are intended to remain consistent with the original Jupyter notebook.

---

## 2. Project Structure

```text
training_pipeline/
│
├── configs/
│   ├── __init__.py
│   └── training_config.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── trainer.py
│   └── train.py
│
├── outputs/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
```text
ingestion_pipeline/
│
|
├── 
│   ├── config.py
│   ├── database.py
│   ├── exporter.py
│   ├── extractor.py
│   └── logger.py
|    |___main.py
|    |___processor.py

```

---

## 3. File Responsibilities

| File | Responsibility |
|---|---|
| `training_config.py` | Stores model, dataset and training configurations. |
| `data.py` | Loads, selects, shuffles, splits and formats the dataset. |
| `model.py` | Loads the Qwen model and tokenizer and attaches LoRA adapters. |
| `trainer.py` | Builds the Hugging Face SFTTrainer and training arguments. |
| `train.py` | Orchestrates the complete training workflow. |
| `requirements.txt` | Lists Python dependencies. |
| `.env.example` | Documents required environment variables. |
| `.gitignore` | Excludes secrets, checkpoints and generated artifacts from Git. |
|`beam_app.py` |initiate the traning on beam cloud 
|`beam_config.py` |use to config traning config on the cloud

---

## 4. Installation

Python dependencies can be installed using:

```bash
pip install -r requirements.txt
```

A compatible PyTorch, CUDA and BitsAndBytes environment is required for GPU training.

The final cloud execution environment will be configured separately using Beam.

---

## 5. Environment Variables

Create a `.env` file from `.env.example` when local environment configuration is implemented.

Required Hugging Face credential:

```env
REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
BUCKET_NAME = 
HF_TOKEN=
ARN=
MONGODB_URI=
DATABASE_NAME=
COLLECTION_NAME=

# # Hugging Face access token
# HF_TOKEN=your_huggingface_token_here

# Model and dataset
HF_MODEL_NAME=Qwen/Qwen2-1.5B-Instruct
HF_DATASET_NAME=KisanVaani/agriculture-qa-english-only

# beam
BEAM_TOKEN=

```

Never commit real credentials to GitHub.

For cloud execution, credentials should be supplied through Beam's secret management.

---

## 6. Training Workflow

The pipeline follows these steps:

1. Load the agricultural dataset from Hugging Face.
2. Shuffle the dataset using seed 42.
3. Select 8,000 samples.
4. Split the selected samples into training and validation sets.
5. Load the Qwen 2 1.5B Instruct model.
6. Apply 4-bit NF4 quantization.
7. Configure and attach LoRA adapters.
8. Initialize the Hugging Face SFTTrainer.
9. Execute supervised fine-tuning.
10. Save the trained adapter, tokenizer and trainer state.

---

## 7. Running Training

The intended entry point is:

```bash
python -m src.train
```

Training execution will be enabled after dependency compatibility checks and cloud environment configuration are completed.

Do not launch the training job until those checks have passed.

---

## 8. Model Output

After a successful training run, the expected output structure is:

```text
outputs/
│
├── checkpoints/
│
└── qwen_agriculture_lora/
    │
    ├── adapter/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    │
    ├── tokenizer/
    │
    └── trainer_state.json
```

The adapter contains the fine-tuned LoRA weights.

The base Qwen model is still required when loading the adapter for inference.

---

## 9. Cloud Training

Beam will be used as the cloud execution platform.

The intended cloud workflow is:

1. Prepare the Beam execution environment.
2. Install compatible training dependencies.
3. Provide Hugging Face credentials securely.
4. Provision the required GPU.
5. Execute `python -m training_pipeline/cloud/beam_app.py`.
6. Persist checkpoints and final model artifacts.
7. Retrieve the trained artifacts.

Cloud execution and persistent storage configuration are not yet implemented.

---

## 10.Beam cloud traning setup
- create account on beam cloud: https://www.beam.cloud/
-create an api_key and save it in the env file
-install the beam sdk using ` pip install beam-client`
- authenticate your local workspace to have access to remote workspace using : `beam configure default --token YOUR_BEAM_TOKEN`



---

