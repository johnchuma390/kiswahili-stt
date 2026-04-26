from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import json
from jiwer import wer, cer

print("Loading test set...")
test_set = load_from_disk("data/splits/test")
print(f"Test samples: {len(test_set)}")

print("\nLoading fine-tuned model...")
processor = WhisperProcessor.from_pretrained("models/finetuned")
model = WhisperForConditionalGeneration.from_pretrained("models/finetuned")
model.eval()

references = []
hypotheses = []
errors = 0

print("\nRunning transcription on test set...")
for i, sample in enumerate(test_set):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(test_set)}")
    try:
        audio = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        reference = sample["transcription"].strip()

        inputs = processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                language="sw",
                task="transcribe"
            )

        hypothesis = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].strip().lower()

        references.append(reference)
        hypotheses.append(hypothesis)

    except Exception as e:
        errors += 1
        print(f"  Error on sample {i}: {e}")

print(f"\nTranscription complete. Errors: {errors}")

finetuned_wer = wer(references, hypotheses)
finetuned_cer = cer(references, hypotheses)

# Load baseline results for comparison
with open("results/baseline_results.json") as f:
    baseline = json.load(f)

print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"{'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Improvement':>12}")
print("-"*60)
print(f"{'Word Error Rate':<25} {baseline['wer_percent']:>11.2f}% {finetuned_wer*100:>11.2f}% {baseline['wer_percent'] - finetuned_wer*100:>11.2f}%")
print(f"{'Char Error Rate':<25} {baseline['cer_percent']:>11.2f}% {finetuned_cer*100:>11.2f}% {baseline['cer_percent'] - finetuned_cer*100:>11.2f}%")
print("="*60)

print("\nSample predictions (first 5):")
for i in range(5):
    print(f"\n  Sample {i+1}:")
    print(f"  REF       : {references[i]}")
    print(f"  FINETUNED : {hypotheses[i]}")

results = {
    "model": "openai/whisper-small fine-tuned on FLEURS sw_ke",
    "stage": "finetuned_round1",
    "dataset": "FLEURS sw_ke test set",
    "samples_evaluated": len(references),
    "word_error_rate": round(finetuned_wer, 4),
    "char_error_rate": round(finetuned_cer, 4),
    "wer_percent": round(finetuned_wer * 100, 2),
    "cer_percent": round(finetuned_cer * 100, 2),
    "baseline_wer_percent": baseline["wer_percent"],
    "baseline_cer_percent": baseline["cer_percent"],
    "wer_improvement": round(baseline["wer_percent"] - finetuned_wer * 100, 2),
    "cer_improvement": round(baseline["cer_percent"] - finetuned_cer * 100, 2)
}

with open("results/finetuned_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results/finetuned_results.json")
print("Evaluation complete.")
