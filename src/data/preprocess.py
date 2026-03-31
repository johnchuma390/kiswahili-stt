from datasets import load_from_disk, DatasetDict
import re
import numpy as np

print("Loading FLEURS dataset...")
dataset = load_from_disk("data/raw/fleurs_sw")

print(f"Before cleaning:")
for split in dataset:
    print(f"  {split}: {len(dataset[split])} samples")

def is_valid_sample(sample):
    audio = sample["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    if duration > 30.0:
        return False
    if duration < 1.0:
        return False
    if len(sample["transcription"].strip()) < 2:
        return False
    return True

def clean_transcription(sample):
    text = sample["transcription"]
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\'\-\.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    sample["transcription"] = text
    return sample

print("\nFiltering samples longer than 30s or shorter than 1s...")
filtered = DatasetDict()
for split in dataset:
    before = len(dataset[split])
    filtered[split] = dataset[split].filter(
        is_valid_sample,
        desc=f"Filtering {split}"
    )
    after = len(filtered[split])
    removed = before - after
    print(f"  {split}: {before} -> {after} samples ({removed} removed)")

print("\nCleaning transcriptions...")
cleaned = DatasetDict()
for split in filtered:
    cleaned[split] = filtered[split].map(
        clean_transcription,
        desc=f"Cleaning {split}"
    )

print("\nAfter cleaning:")
for split in cleaned:
    print(f"  {split}: {len(cleaned[split])} samples")

durations = []
for sample in cleaned["train"]:
    duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
    durations.append(duration)
durations = np.array(durations)
print(f"\nCleaned training set duration stats:")
print(f"  Total samples : {len(durations)}")
print(f"  Min duration  : {durations.min():.2f} sec")
print(f"  Max duration  : {durations.max():.2f} sec")
print(f"  Mean duration : {durations.mean():.2f} sec")
print(f"  Total audio   : {durations.sum()/3600:.2f} hours")

print("\nSample cleaned transcriptions:")
for i in range(5):
    print(f"  {i+1}. {cleaned['train'][i]['transcription']}")

print("\nSaving cleaned dataset to data/processed/fleurs_sw_cleaned...")
cleaned.save_to_disk("data/processed/fleurs_sw_cleaned")
print("Done.")
