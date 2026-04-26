import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Kiswahili ASR.")
    parser.add_argument("--model-name", default="openai/whisper-small", help="Base Whisper model")
    parser.add_argument("--train-dataset", default="data/splits/train", help="Path to train split")
    parser.add_argument("--validation-dataset", default="data/splits/validation", help="Path to validation split")
    parser.add_argument("--output-dir", default="models/checkpoints/whisper-small-sw-ft", help="Checkpoint output dir")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=float, default=10.0, help="Training epochs")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--eval-steps", type=int, default=150, help="Evaluation interval")
    parser.add_argument("--save-steps", type=int, default=150, help="Checkpoint save interval")
    parser.add_argument("--max-label-length", type=int, default=225, help="Max tokenizer label length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds = load_from_disk(args.train_dataset)
    val_ds = load_from_disk(args.validation_dataset)

    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=16000))

    print("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="sw", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    model.generation_config.language = "sw"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="sw", task="transcribe")

    if model.config.decoder_start_token_id is None:
        raise ValueError("Model is missing decoder_start_token_id")

    def prepare_batch(sample: Dict[str, Any]) -> Dict[str, Any]:
        audio = sample["audio"]
        sample["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        sample["labels"] = processor.tokenizer(
            sample["transcription"],
            max_length=args.max_label_length,
            truncation=True,
        ).input_ids
        return sample

    print("Preparing train features...")
    train_prepared = train_ds.map(
        prepare_batch,
        remove_columns=train_ds.column_names,
        desc="Preparing train set",
    )

    print("Preparing validation features...")
    val_prepared = val_ds.map(
        prepare_batch,
        remove_columns=val_ds.column_names,
        desc="Preparing validation set",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [p.strip().lower() for p in pred_str]
        label_str = [l.strip().lower() for l in label_str]

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    use_cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=8 if use_cuda else 2,
        per_device_eval_batch_size=8 if use_cuda else 2,
        gradient_accumulation_steps=2 if use_cuda else 8,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=-1,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=use_cuda,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        weight_decay=args.weight_decay,
        save_total_limit=2,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_prepared,
        eval_dataset=val_prepared,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metrics = trainer.evaluate(metric_key_prefix="eval")
    metrics["train_runtime_sec"] = round(train_result.metrics.get("train_runtime", 0.0), 2)
    metrics["train_loss"] = round(train_result.metrics.get("train_loss", 0.0), 4)

    results_path = Path("results/finetuned_results.json")
    payload = {
        "model": args.model_name,
        "output_dir": str(output_dir),
        "stage": "finetuned_validation",
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved model and processor to {output_dir}")
    print(f"Saved validation metrics to {results_path}")


if __name__ == "__main__":
    main()
