from datasets import load_from_disk
import json

print("Loading cleaned dataset...")
dataset = load_from_disk("data/processed/fleurs_sw_cleaned")

print("\nFinal split summary:")
total = 0
summary = {}
for split in dataset:
    count = len(dataset[split])
    total += count
    summary[split] = count
    print(f"  {split}: {count} samples")
print(f"  TOTAL : {total} samples")

print("\nSaving splits to data/splits/...")
dataset["train"].save_to_disk("data/splits/train")
dataset["validation"].save_to_disk("data/splits/validation")
dataset["test"].save_to_disk("data/splits/test")

stats = {
    "dataset": "FLEURS Swahili (sw_ke)",
    "splits": summary,
    "total_samples": total,
    "train_hours": 12.75,
    "sampling_rate": 16000,
    "max_duration_sec": 30,
    "min_duration_sec": 1,
    "preprocessing": [
        "Filtered samples > 30 seconds",
        "Filtered samples < 1 second",
        "Lowercased transcriptions",
        "Removed special characters",
        "Normalised whitespace"
    ]
}

with open("results/dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\nDataset stats saved to results/dataset_stats.json")

print("\nAll splits saved. Data preparation complete.")
