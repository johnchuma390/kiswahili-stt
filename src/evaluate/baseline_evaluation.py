from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import json
from jiwer import wer, cer

print("Loading test set...")
test_set = load_from_disk("data/splits/test")
print(f"Test samples: {len(test_set)}")

print("\nLoading Whisper-Small model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
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

baseline_wer = wer(references, hypotheses)
baseline_cer = cer(references, hypotheses)

print("\n" + "="*60)
print("BASELINE RESULTS (Whisper-Small, zero-shot)")
print("="*60)
print(f"  Samples evaluated : {len(references)}")
print(f"  Word Error Rate   : {baseline_wer:.4f} ({baseline_wer*100:.2f}%)")
print(f"  Char Error Rate   : {baseline_cer:.4f} ({baseline_cer*100:.2f}%)")
print("="*60)

print("\nSample predictions (first 5):")
for i in range(5):
    print(f"\n  Sample {i+1}:")
    print(f"  REF : {references[i]}")
    print(f"  HYP : {hypotheses[i]}")

results = {
    "model": "openai/whisper-small",
    "stage": "baseline_zero_shot",
    "dataset": "FLEURS sw_ke test set",
    "samples_evaluated": len(references),
    "word_error_rate": round(baseline_wer, 4),
    "char_error_rate": round(baseline_cer, 4),
    "wer_percent": round(baseline_wer * 100, 2),
    "cer_percent": round(baseline_cer * 100, 2)
}

with open("results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results/baseline_results.json")
print("Baseline evaluation complete.")
