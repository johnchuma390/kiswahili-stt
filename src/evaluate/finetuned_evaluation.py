import argparse
import json
from pathlib import Path
from typing import List

import torch
from datasets import Audio, load_from_disk
from jiwer import cer, wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper checkpoint.")
    parser.add_argument("--checkpoint", default="models/checkpoints/whisper-small-sw-ft", help="Path to fine-tuned model")
    parser.add_argument("--test-dataset", default="data/splits/test", help="Path to test split")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for batched generation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output", default="results/finetuned_results.json", help="Where to write evaluation JSON")
    return parser.parse_args()


def batched(items: List[dict], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main() -> None:
    args = parse_args()
    Path("results").mkdir(parents=True, exist_ok=True)

    print(f"Loading test set from: {args.test_dataset}")
    test_set = load_from_disk(args.test_dataset).cast_column("audio", Audio(sampling_rate=16000))
    print(f"Test samples: {len(test_set)}")

    print(f"Loading checkpoint from: {args.checkpoint}")
    processor = WhisperProcessor.from_pretrained(args.checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()

    model.generation_config.language = "sw"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="sw", task="transcribe")

    references = []
    hypotheses = []
    errors = 0

    records = list(test_set)
    for i, chunk in enumerate(batched(records, args.batch_size)):
        if i % 10 == 0:
            done = min(i * args.batch_size, len(records))
            print(f"  Progress: {done}/{len(records)}")

        try:
            audio_arrays = [sample["audio"]["array"] for sample in chunk]
            refs = [sample["transcription"].strip().lower() for sample in chunk]

            inputs = processor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            input_features = inputs["input_features"].to(args.device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features)

            preds = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            preds = [p.strip().lower() for p in preds]

            references.extend(refs)
            hypotheses.extend(preds)
        except Exception as exc:
            errors += len(chunk)
            print(f"  Batch failed: {exc}")

    test_wer = wer(references, hypotheses)
    test_cer = cer(references, hypotheses)

    print("\n" + "=" * 60)
    print("FINETUNED RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated : {len(references)}")
    print(f"  Errors            : {errors}")
    print(f"  Word Error Rate   : {test_wer:.4f} ({test_wer * 100:.2f}%)")
    print(f"  Char Error Rate   : {test_cer:.4f} ({test_cer * 100:.2f}%)")
    print("=" * 60)

    if references:
        print("\nSample predictions (first 5):")
        for idx in range(min(5, len(references))):
            print(f"\n  Sample {idx + 1}:")
            print(f"  REF : {references[idx]}")
            print(f"  HYP : {hypotheses[idx]}")

    results = {
        "model": str(args.checkpoint),
        "stage": "finetuned_test",
        "dataset": "FLEURS sw_ke test set",
        "samples_evaluated": len(references),
        "errors": errors,
        "word_error_rate": round(test_wer, 4),
        "char_error_rate": round(test_cer, 4),
        "wer_percent": round(test_wer * 100, 2),
        "cer_percent": round(test_cer * 100, 2),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
