from datasets import load_from_disk
import librosa
import numpy as np
import matplotlib.pyplot as plt

print("Loading FLEURS dataset from disk...")
dataset = load_from_disk("data/raw/fleurs_sw")

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
for split in dataset:
    print(f"  {split}: {len(dataset[split])} samples")

print("\n" + "="*60)
print("COLUMN NAMES")
print("="*60)
print(dataset["train"].column_names)

print("\n" + "="*60)
print("FIRST 3 TRAINING SAMPLES")
print("="*60)
for i in range(3):
    sample = dataset["train"][i]
    audio = sample["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    print(f"\nSample {i+1}:")
    print(f"  Transcription : {sample['transcription']}")
    print(f"  Sampling rate : {audio['sampling_rate']} Hz")
    print(f"  Duration      : {duration:.2f} seconds")
    print(f"  Audio shape   : {np.array(audio['array']).shape}")

print("\n" + "="*60)
print("AUDIO DURATION STATISTICS (training set)")
print("="*60)
durations = []
sampling_rates = set()
for sample in dataset["train"]:
    audio = sample["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    durations.append(duration)
    sampling_rates.add(audio["sampling_rate"])

durations = np.array(durations)
print(f"  Total samples      : {len(durations)}")
print(f"  Min duration       : {durations.min():.2f} sec")
print(f"  Max duration       : {durations.max():.2f} sec")
print(f"  Mean duration      : {durations.mean():.2f} sec")
print(f"  Median duration    : {np.median(durations):.2f} sec")
print(f"  Total audio        : {durations.sum()/3600:.2f} hours")
print(f"  Sampling rates     : {sampling_rates}")

print("\n" + "="*60)
print("TRANSCRIPTION SAMPLES (first 10)")
print("="*60)
for i in range(10):
    print(f"  {i+1}. {dataset['train'][i]['transcription']}")

print("\n" + "="*60)
print("TRANSCRIPTION LENGTH STATISTICS")
print("="*60)
lengths = [len(dataset["train"][i]["transcription"].split())
           for i in range(len(dataset["train"]))]
lengths = np.array(lengths)
print(f"  Min words  : {lengths.min()}")
print(f"  Max words  : {lengths.max()}")
print(f"  Mean words : {lengths.mean():.1f}")

print("\nExploration complete.")
