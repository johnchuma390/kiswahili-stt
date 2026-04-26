import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from datasets import Audio, concatenate_datasets, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an augmented train split by adding Common Voice Swahili to existing FLEURS splits."
    )
    parser.add_argument("--base-splits-dir", default="data/splits", help="Directory containing train/validation/test splits")
    parser.add_argument("--common-voice-dir", default="data/raw/common_voice_sw", help="Path to Common Voice dataset on disk")
    parser.add_argument("--output-dir", default="data/splits_augmented", help="Output directory for augmented splits")
    parser.add_argument("--min-duration", type=float, default=1.0, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum clip duration in seconds")
    parser.add_argument("--max-cv-samples", type=int, default=0, help="Optional cap for Common Voice samples (0 means no cap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s'\-.]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def clean_common_voice(example: Dict) -> Dict:
    example["transcription"] = normalize_text(example.get("sentence", ""))
    return example


def is_valid(example: Dict, min_duration: float, max_duration: float) -> bool:
    audio = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    text = example.get("transcription", "").strip()
    return min_duration <= duration <= max_duration and len(text) >= 2


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    print("Loading base splits...")
    train_base = load_from_disk(str(Path(args.base_splits_dir) / "train")).cast_column("audio", Audio(sampling_rate=16000))
    val_base = load_from_disk(str(Path(args.base_splits_dir) / "validation")).cast_column("audio", Audio(sampling_rate=16000))
    test_base = load_from_disk(str(Path(args.base_splits_dir) / "test")).cast_column("audio", Audio(sampling_rate=16000))

    print("Loading Common Voice Swahili...")
    common_voice = load_from_disk(args.common_voice_dir)

    cv_parts: List = []
    for split_name in ["train", "validation"]:
        if split_name not in common_voice:
            continue
        current = common_voice[split_name].cast_column("audio", Audio(sampling_rate=16000))
        current = current.map(clean_common_voice, desc=f"Cleaning Common Voice {split_name}")
        current = current.filter(
            lambda x: is_valid(x, args.min_duration, args.max_duration),
            desc=f"Filtering Common Voice {split_name}",
        )
        current = current.remove_columns([c for c in current.column_names if c not in {"audio", "transcription"}])
        cv_parts.append(current)

    if not cv_parts:
        raise RuntimeError("No Common Voice train/validation splits found. Cannot build augmented training set.")

    cv_combined = concatenate_datasets(cv_parts)
    if args.max_cv_samples > 0 and len(cv_combined) > args.max_cv_samples:
        cv_combined = cv_combined.shuffle(seed=args.seed).select(range(args.max_cv_samples))

    print(f"Base train samples       : {len(train_base)}")
    print(f"Common Voice extra train : {len(cv_combined)}")

    augmented_train = concatenate_datasets([train_base, cv_combined]).shuffle(seed=args.seed)

    print(f"Augmented train samples  : {len(augmented_train)}")
    print(f"Validation samples       : {len(val_base)}")
    print(f"Test samples             : {len(test_base)}")

    augmented_train.save_to_disk(str(output_dir / "train"))
    val_base.save_to_disk(str(output_dir / "validation"))
    test_base.save_to_disk(str(output_dir / "test"))

    summary = {
        "base_train_samples": len(train_base),
        "common_voice_added_samples": len(cv_combined),
        "augmented_train_samples": len(augmented_train),
        "validation_samples": len(val_base),
        "test_samples": len(test_base),
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "max_cv_samples": args.max_cv_samples,
        "output_dir": str(output_dir),
    }

    with open("results/augmented_dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved augmented splits to:", output_dir)
    print("Saved dataset summary to results/augmented_dataset_stats.json")


if __name__ == "__main__":
    main()
