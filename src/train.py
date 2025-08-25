import os, re, json, argparse, soundfile as sf, numpy as np, pandas as pd
from datasets import Dataset
import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
from jiwer import wer

def build_vocab(vocab_path="vocab.json"):
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    vocab_dict = {c: i for i, c in enumerate(alphabet)}
    vocab_dict["|"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)
    return vocab_dict

def load_dataset(data_path):
    train_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    test_df  = pd.read_csv(os.path.join(data_path, "test_labels.csv"))
    train_df["path"] = train_df["id"].apply(lambda x: os.path.join(data_path, "train/audio", x + ".wav"))
    test_df["path"]  = test_df["id"].apply(lambda x: os.path.join(data_path, "test/audio", x + ".wav"))
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

def prepare_batch(batch, processor):
    audio_array, sr = sf.read(batch["path"])
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    batch["input_values"] = processor(audio_array, sampling_rate=16000).input_values[0]
    spaced = " ".join(list(batch["text"].upper()))
    with processor.as_target_processor():
        batch["labels"] = processor(spaced).input_ids
    return batch

def compute_metrics(pred, processor):
    pred_ids = torch.argmax(torch.tensor(pred.predictions), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    with processor.as_target_processor():
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer(label_str, pred_str)}

def main(args):
    vocab_dict = build_vocab()
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    train_ds, test_ds = load_dataset(args.data_path)
    train_ds = train_ds.map(lambda b: prepare_batch(b, processor))
    test_ds  = test_ds.map(lambda b: prepare_batch(b, processor))
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h",
        vocab_size=len(processor.tokenizer.get_vocab()),
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"✅ Training done. Model + processor saved at {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="captcha_dataset")
    parser.add_argument("--output_dir", type=str, default="wav2vec2-captcha")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    main(args)
