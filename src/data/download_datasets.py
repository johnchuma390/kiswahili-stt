from datasets import load_dataset
import os

print("="*60)
print("Step 1: Downloading FLEURS Swahili...")
print("="*60)

fleurs = load_dataset(
    "google/fleurs",
    "sw_ke",
    trust_remote_code=True
)

print("\nFLEURS splits and sizes:")
for split in fleurs:
    print(f"  {split}: {len(fleurs[split])} samples")

fleurs.save_to_disk("data/raw/fleurs_sw")
print("FLEURS saved to data/raw/fleurs_sw")

print("\n" + "="*60)
print("Step 2: Downloading Common Voice 17 Swahili...")
print("="*60)

common_voice = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "sw",
    trust_remote_code=True
)

print("\nCommon Voice splits and sizes:")
for split in common_voice:
    print(f"  {split}: {len(common_voice[split])} samples")

common_voice.save_to_disk("data/raw/common_voice_sw")
print("Common Voice saved to data/raw/common_voice_sw")

print("\n" + "="*60)
print("All datasets downloaded successfully")
print("="*60)
