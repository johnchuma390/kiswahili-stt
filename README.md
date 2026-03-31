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

## Academic Year
2025/2026 — Final Year Project
