# Kiswahili Speech-to-Text System for Education

A lightweight deep learning system that converts spoken Kiswahili into text,
designed to run on low-cost hardware (Android/Raspberry Pi) for use in
Kenyan classrooms.

## Project Structure
```
kiswahili-stt/
├── data/
│   ├── raw/          # Downloaded datasets (not committed to Git)
│   ├── processed/    # Cleaned and resampled audio
│   └── splits/       # Train/validation/test splits
├── notebooks/        # Jupyter notebooks for exploration and training
├── src/
│   ├── data/         # Data loading and preprocessing scripts
│   ├── model/        # Model loading and fine-tuning scripts
│   ├── evaluate/     # Evaluation and metrics scripts
│   └── app/          # Gradio demo application
├── models/
│   ├── checkpoints/  # Training checkpoints (not committed to Git)
│   └── quantised/    # Optimised models for edge deployment
├── results/          # Evaluation results and metrics logs
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training Workflow

1. Download and preprocess data:
```bash
python src/data/download_datasets.py
python src/data/preprocess.py
python src/data/prepare_splits.py
```

2. Run baseline evaluation:
```bash
python src/evaluate/baseline_evaluation.py
```

3. Fine-tune Whisper on Kiswahili:
```bash
python src/model/finetune_whisper.py \
	--model-name openai/whisper-small \
	--output-dir models/checkpoints/whisper-small-sw-ft \
	--learning-rate 1e-5 \
	--epochs 10
```

4. Evaluate fine-tuned checkpoint on test set:
```bash
python src/evaluate/finetuned_evaluation.py \
	--checkpoint models/checkpoints/whisper-small-sw-ft \
	--output results/finetuned_results.json
```

## Fine-Tune With More Data

To increase training data, build an augmented train split by adding cleaned Common Voice Swahili clips to your current FLEURS train split:

```bash
python src/data/prepare_augmented_train.py \
	--base-splits-dir data/splits \
	--common-voice-dir data/raw/common_voice_sw \
	--output-dir data/splits_augmented
```

Then fine-tune on the augmented split:

```bash
python src/model/finetune_whisper.py \
	--train-dataset data/splits_augmented/train \
	--validation-dataset data/splits_augmented/validation \
	--output-dir models/checkpoints/whisper-small-sw-ft-augmented \
	--learning-rate 1e-5 \
	--epochs 12
```

Finally evaluate on the unchanged test split for fair comparison:

```bash
python src/evaluate/finetuned_evaluation.py \
	--checkpoint models/checkpoints/whisper-small-sw-ft-augmented \
	--test-dataset data/splits_augmented/test \
	--output results/finetuned_results_augmented.json
```

## Tips To Improve Results

- Increase effective batch size using gradient accumulation if GPU memory is limited.
- Try learning rates in the range 5e-6 to 2e-5 and compare validation WER.
- Train for 10-20 epochs, but keep the best checkpoint based on validation WER.
- Keep language/task prompts fixed to Swahili during generation for stable decoding.
- Compare both WER and CER across runs in `results/finetuned_results.json`.

## Academic Year
2025/2026 — Final Year Project
