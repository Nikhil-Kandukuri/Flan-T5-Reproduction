import torch
import datasets
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, TrainingArguments
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wandb
import argparse

def main():

    wandb.login(key='b371339878dcde7d6a8fa36cca8f0a2433c79229') 

    # dataset_1 = load_dataset("DataProvenanceInitiative/t0_submix_original",)
    # dataset_2 = load_dataset("DataProvenanceInitiative/dialog_submix_original")
    # dataset_3 = load_dataset("DataProvenanceInitiative/niv2_submix_original")
    # dataset_4 = load_dataset("DataProvenanceInitiative/cot_submix_original")
    # dataset_5 = load_dataset("DataProvenanceInitiative/flan2021_submix_original")

    dataset_1 = load_dataset("/data/tir/projects/tir6/general/nkanduku/.cache/huggingface/datasets/DataProvenanceInitiative___cot_submix_original")
    dataset_2 = load_dataset("/data/tir/projects/tir6/general/nkanduku/.cache/huggingface/datasets/DataProvenanceInitiative___dialog_submix_original")
    dataset_3 = load_dataset("/data/tir/projects/tir6/general/nkanduku/.cache/huggingface/datasets/DataProvenanceInitiative___niv2_submix_original")
    dataset_4 = load_dataset("/data/tir/projects/tir6/general/nkanduku/.cache/huggingface/datasets/DataProvenanceInitiative___cot_submix_original")
    dataset_5 = load_dataset("/data/tir/projects/tir6/general/nkanduku/.cache/huggingface/datasets/DataProvenanceInitiative___flan2021_submix_original")



    dataset = datasets.concatenate_datasets([dataset_1["train"],dataset_2["train"], dataset_3["train"],dataset_4["train"],dataset_5["train"]])

    # dataset = datasets.concatenate_datasets([dataset_1["train"],dataset_2["train"]])
    # raise ValueError("Stop")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/t5-large-lm-adapt",use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-large-lm-adapt")
    print(f"Number of parameters: {model.num_parameters():,}")

    # # Preprocess data
    def preprocess_data(examples):
        inputs = examples["inputs"]
        targets = examples["targets"]
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_data, batched=True)
    train_dataset = tokenized_datasets
    val_dataset = train_dataset.select(range(1000))

    # Set up data collator and configs
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(report_to="wandb",save_steps=40000,output_dir=args.output_dir, 
                                            num_train_epochs=args.num_train_epochs,per_device_train_batch_size=args.per_device_train_batch_size,learning_rate=args.learning_rate)  


    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    start_time = time.time()

    # Train model
    trainer.train()

    end_time = time.time()

    # Calculate the total execution time
    total_time = end_time - start_time
    print("Total time taken to train one epoch",total_time)
    print(f"Total execution time: {total_time} seconds")

    # Evaluate model
    eval_result = trainer.evaluate()

    # Save model 
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Train T5 model')
    parser.add_argument('--output_dir', type=str, help='Directory to save the trained model')
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the optimizer.")
    args = parser.parse_args()
    main()
