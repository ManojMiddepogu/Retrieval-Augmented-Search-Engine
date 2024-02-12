import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
from torch import bfloat16, LongTensor, cuda
import random
import bitsandbytes as bnb

def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )


wandb.init(
    # set the wandb project where this run will be logged
    project="dfs_searching",
    entity="mm12799",
    name="mistral_model"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model_id = "meta-llama/Llama-2-7b-chat-hf"
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
    use_cache=False,
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
#--------------------------------------------------------
#--------------------Training----------------------------

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./llm/"
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="wandb"
)

prompt = """<s> [INST] You are an assistant for search tasks.
Use the following pieces of retrieved context and your knowledge to answer the search query.
Please provide a direct response to the question without additional comments on the context.
[/INST]
Search Query: {question}
Context: {context}
Answer: {response} </s>
"""

def formatting_prompts_func(example):
    text = prompt.format(
        context=example['context'],
        question=example['question'],
        response=example['response']
    )
    return text

# dataset = load_dataset("csv", data_files="/scratch/mm12799/we_com/train_data.csv",split="train")
dataset = load_dataset("csv", data_files="/scratch/mm12799/we_com/datasets/dolly/dolly.csv",split="train")
print(dataset)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    peft_config = peft_config,
    train_dataset=dataset,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
    formatting_func=formatting_prompts_func
)
trainer.train()

# Fine-tuned model name
# path = "./results/llama-2-7b-chat-we-com3"
path = "./results/mistral-we-com3"
trainer.model.save_pretrained(path)
tokenizer.save_pretrained(path)