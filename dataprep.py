"""
Script to prepare your dataset for VITS/MMS finetuning.
Converts your CSV format to HuggingFace datasets format.
"""

import pandas as pd
from datasets import Dataset, Audio, DatasetDict
import os
import argparse
import json

# Language code mapping
LANGUAGE_CODES = {
    'yoruba': 'yor',
    'pidgin': 'pcm',
    'french': 'fra',
    'hausa': 'hau',
    'swahili': 'swh',
    'kinyarwanda': 'kin',
    'zulu': 'zul'
}


def create_config_for_language(
    language: str,
    dataset_path: str,
    output_dir: str,
    model_output_dir: str,
    checkpoint_path: str,
    hub_model_id: str = None,
    speaker_id_to_use: str = None,
    num_epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """
    Create a training config JSON for a specific language.
    """
    language_lower = language.lower()
    lang_code = LANGUAGE_CODES.get(language_lower)
    
    if not lang_code:
        raise ValueError(f"Language {language} not found in mapping")
    
    # Default model name
    if hub_model_id is None:
        hub_model_id = f"mms-tts-{lang_code}-finetuned"
    
    config = {
        "project_name": f"mms_{language_lower}_finetuning",
        "push_to_hub": True,
        "hub_model_id": hub_model_id,
        "report_to": ["tensorboard"],
        "overwrite_output_dir": True,
        "output_dir": model_output_dir,
        
        # Dataset config
        "dataset_name": dataset_path,
        "audio_column_name": "audio",
        "text_column_name": "text",
        "train_split_name": "train",
        "eval_split_name": "validation",
        "speaker_id_column_name": "speaker_id",
        "override_speaker_embeddings": True,
        
        # Audio filtering
        "max_duration_in_seconds": 20,
        "min_duration_in_seconds": 1.0,
        "max_tokens_length": 500,
        
        # Model checkpoint
        "model_name_or_path": checkpoint_path,
        
        # Training hyperparameters
        "preprocessing_num_workers": 4,
        "do_train": True,
        "num_train_epochs": num_epochs,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "adam_beta1": 0.8,
        "adam_beta2": 0.99,
        "warmup_ratio": 0.01,
        "group_by_length": False,
        
        # Evaluation
        "do_eval": True,
        "eval_steps": 50,
        "per_device_eval_batch_size": batch_size,
        "max_eval_samples": 25,
        "do_step_schedule_per_epoch": True,
        
        # Loss weights
        "weight_disc": 3,
        "weight_fmaps": 1,
        "weight_gen": 1,
        "weight_kl": 1.5,
        "weight_duration": 1,
        "weight_mel": 35,
        
        "fp16": True,
        "seed": 456
    }
    
    # If single speaker model
    if speaker_id_to_use is not None:
        config["filter_on_speaker_id"] = speaker_id_to_use
    
    # Save config
    config_path = os.path.join(output_dir, f"finetune_{language_lower}_config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset and config for VITS/MMS finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_vits_dataset.py \\
      --csv_path data.csv \\
      --audio_base_path /path/to/audio \\
      --language yoruba
  
  # High-quality single speaker
  python prepare_vits_dataset.py \\
      --csv_path data.csv \\
      --audio_base_path /path/to/audio \\
      --language french \\
      --speaker_id 42 \\
      --min_snr 25.0
  
  # Custom training parameters
  python prepare_vits_dataset.py \\
      --csv_path data.csv \\
      --audio_base_path /path/to/audio \\
      --language hausa \\
      --num_epochs 150 \\
      --batch_size 32 \\
      --learning_rate 1e-5
"""
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to your CSV file with metadata"
    )
    required.add_argument(
        "--audio_base_path",
        type=str,
        required=True,
        help="Base path where audio files are stored"
    )
    required.add_argument(
        "--language",
        type=str,
        required=True,
        choices=list(LANGUAGE_CODES.keys()),
        help="Target language to prepare"
    )
    
    # Output paths
    paths = parser.add_argument_group('output paths')
    paths.add_argument(
        "--dataset_output_dir",
        type=str,
        default=None,
        help="Where to save the prepared dataset (default: ./datasets/{language})"
    )
    paths.add_argument(
        "--config_output_dir",
        type=str,
        default="./training_configs",
        help="Where to save training config (default: ./training_configs)"
    )
    paths.add_argument(
        "--model_output_dir",
        type=str,
        default=None,
        help="Where to save trained model (default: ./models/{language}_finetuned)"
    )
    paths.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (default: mms-tts-{lang_code}-finetuned)"
    )
    paths.add_argument(
        "--converted_checkpoint_path",
        type=str,
        default=None,
        help="Path to converted checkpoint (default: ./checkpoints/mms-{lang_code}-train)"
    )
    
    # Data filtering
    filtering = parser.add_argument_group('data filtering')
    filtering.add_argument(
        "--train_split",
        type=float,
        default=0.95,
        help="Proportion of data for training (default: 0.95)"
    )
    filtering.add_argument(
        "--max_duration",
        type=float,
        default=20.0,
        help="Maximum audio duration in seconds (default: 20.0)"
    )
    filtering.add_argument(
        "--min_duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds (default: 0.5)"
    )
    filtering.add_argument(
        "--min_snr",
        type=float,
        default=None,
        help="Minimum SNR in dB to filter by (optional, filters noisy audio)"
    )
    filtering.add_argument(
        "--speaker_id",
        type=str,
        default=None,
        help="Filter by specific speaker ID for single-speaker model (optional)"
    )
    
    # Training parameters
    training = parser.add_argument_group('training parameters')
    training.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)"
    )
    training.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    training.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    
    # Flags
    flags = parser.add_argument_group('control flags')
    flags.add_argument(
        "--skip_config",
        action="store_true",
        help="Skip config generation, only prepare dataset"
    )
    flags.add_argument(
        "--skip_dataset",
        action="store_true",
        help="Skip dataset preparation, only generate config"
    )
    
    args = parser.parse_args()
    
    # Set defaults based on language
    target_language = args.language
    lang_code = LANGUAGE_CODES[target_language]
    
    if args.dataset_output_dir is None:
        args.dataset_output_dir = f"./datasets/{target_language}"
    
    if args.model_output_dir is None:
        args.model_output_dir = f"./models/{target_language}_finetuned"
    
    if args.hub_model_id is None:
        args.hub_model_id = f"mms-tts-{lang_code}-finetuned"
    
    if args.converted_checkpoint_path is None:
        args.converted_checkpoint_path = f"./checkpoints/mms-{lang_code}-train"
    
    # ========================================================================
    # STEP 1: Prepare dataset
    # ========================================================================
    if not args.skip_dataset:
        print(f"\n{'='*70}")
        print(f"📦 Preparing dataset for {target_language.upper()}")
        print(f"{'='*70}\n")
        
        # Read CSV
        print(f"Reading CSV from: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        print(f"Total rows in CSV: {len(df)}")
        
        # Filter by language
        print(f"\nFiltering by language: {target_language}")
        df = df[df['language'].str.lower() == target_language.lower()]
        print(f"After language filter: {len(df)} samples")
        
        if len(df) == 0:
            print(f"\n❌ ERROR: No samples found for language '{target_language}'")
            print(f"Available languages in CSV: {df['language'].unique().tolist()}")
            return
        
        # Filter by duration
        print(f"\nFiltering by duration: {args.min_duration}s - {args.max_duration}s")
        df = df[(df['duration'] >= args.min_duration) & (df['duration'] <= args.max_duration)]
        print(f"After duration filter: {len(df)} samples")
        
        # Apply SNR filter if specified
        if args.min_snr is not None:
            print(f"\nFiltering by SNR >= {args.min_snr} dB")
            df = df[df['snr_db'] >= args.min_snr]
            print(f"After SNR filter: {len(df)} samples")
        
        # Apply speaker filter if specified
        if args.speaker_id is not None:
            print(f"\nFiltering by speaker_id = {args.speaker_id}")
            df = df[df['speaker_id'] == args.speaker_id]
            print(f"After speaker filter: {len(df)} samples")
        
        if len(df) == 0:
            print(f"\n❌ ERROR: No samples left after filtering!")
            return
        
        print(f"\n✓ Final dataset size: {len(df)} samples")
        print(f"✓ Unique speakers: {df['speaker_id'].nunique()}")
        
        # Create full audio paths
        print(f"\nCreating audio paths with base: {args.audio_base_path}")
        df['audio_full_path'] = df['audio_path'].apply(
            lambda x: os.path.join(args.audio_base_path, x)
        )
        
        # Check if audio files exist
        print("Checking if audio files exist...")
        existing_files = df['audio_full_path'].apply(os.path.exists)
        missing_count = (~existing_files).sum()
        
        if missing_count > 0:
            print(f"\n⚠️  WARNING: {missing_count} audio files not found!")
            print(f"Example missing file: {df[~existing_files]['audio_full_path'].iloc[0]}")
            df = df[existing_files]
            print(f"Continuing with {len(df)} samples with existing files")
        else:
            print(f"✓ All audio files found!")
        
        if len(df) == 0:
            print(f"\n❌ ERROR: No audio files found! Check your --audio_base_path")
            return
        
        # Prepare data in HF format
        print(f"\nPreparing HuggingFace Dataset...")
        data = {
            'audio': df['audio_full_path'].tolist(),
            'text': df['transcription'].tolist(),
            'speaker_id': df['speaker_id'].tolist(),
            'duration': df['duration'].tolist(),
        }
        
        # Create HuggingFace dataset
        dataset = Dataset.from_dict(data)
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
        
        # Split into train/validation
        split_idx = int(len(dataset) * args.train_split)
        train_dataset = dataset.select(range(split_idx))
        eval_dataset = dataset.select(range(split_idx, len(dataset)))
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': eval_dataset
        })
        
        # Save to disk
        print(f"\nSaving dataset to: {args.dataset_output_dir}")
        dataset_dict.save_to_disk(args.dataset_output_dir)
        
        print(f"\n{'='*70}")
        print(f"✅ DATASET PREPARED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"  📁 Location: {args.dataset_output_dir}")
        print(f"  📊 Train samples: {len(train_dataset)}")
        print(f"  📊 Validation samples: {len(eval_dataset)}")
        print(f"\n  👥 Speaker distribution (top 10):")
        speaker_counts = df['speaker_id'].value_counts().head(10)
        for speaker, count in speaker_counts.items():
            print(f"     Speaker {speaker}: {count} samples")
    
    # ========================================================================
    # STEP 2: Create training config
    # ========================================================================
    if not args.skip_config:
        print(f"\n{'='*70}")
        print(f"⚙️  Creating training config")
        print(f"{'='*70}\n")
        
        config_path = create_config_for_language(
            language=target_language,
            dataset_path=args.dataset_output_dir,
            output_dir=args.config_output_dir,
            model_output_dir=args.model_output_dir,
            checkpoint_path=args.converted_checkpoint_path,
            hub_model_id=args.hub_model_id,
            speaker_id_to_use=args.speaker_id,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        
        print(f"✅ Config saved to: {config_path}")
        print(f"\n📝 Training settings:")
        print(f"   • Epochs: {args.num_epochs}")
        print(f"   • Batch size: {args.batch_size}")
        print(f"   • Learning rate: {args.learning_rate}")
        print(f"   • Model checkpoint: {args.converted_checkpoint_path}")
    
    # ========================================================================
    # STEP 3: Print next steps
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"🚀 NEXT STEPS")
    print(f"{'='*70}\n")
    
    checkpoint_exists = os.path.exists(args.converted_checkpoint_path)
    
    if not checkpoint_exists:
        print(f"1️⃣  Convert the discriminator checkpoint:\n")
        print(f"    python convert_original_discriminator_checkpoint.py \\")
        print(f"        --language_code {lang_code} \\")
        print(f"        --pytorch_dump_folder_path {args.converted_checkpoint_path}\n")
        print(f"    Optional: Push to hub with:")
        print(f"        --push_to_hub your-username/mms-tts-{lang_code}-train\n")
        step_num = "2️⃣ "
    else:
        print(f"✅ Checkpoint already exists at: {args.converted_checkpoint_path}\n")
        step_num = "1️⃣ "
    
    if not args.skip_config:
        print(f"{step_num} Start training:\n")
        print(f"    accelerate launch run_vits_finetuning.py \\")
        print(f"        {os.path.join(args.config_output_dir, f'finetune_{target_language}_config.json')}\n")
        print(f"    Or for multi-GPU:")
        print(f"    accelerate launch --multi_gpu run_vits_finetuning.py \\")
        print(f"        {os.path.join(args.config_output_dir, f'finetune_{target_language}_config.json')}\n")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()