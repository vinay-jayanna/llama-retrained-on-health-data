from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoConfig
from datasets import Dataset
import pandas as pd
import kagglehub
from huggingface_hub import login

# Login to Hugging Face Hub
huggingface_token = ""  # Replace with your Hugging Face token
login(huggingface_token)

# Define model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"  # Replace with the LLaMA model of your choice

# Load the config and fix rope_scaling
config = AutoConfig.from_pretrained(model_name)
  # Ensure only required fields exist
config.rope_scaling = {"type": "linear", "factor": 2.0}

# Load model with modified config
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings

# Load and prepare the dataset
path = kagglehub.dataset_download("jpmiller/layoutlm")
data_path = f"{path}/medquad.csv"  # Adjust the CSV file path if needed
df = pd.read_csv(data_path)
df = df.sample(frac=0.1)  # Take 10% of the dataset

print("Number of rows in the DataFrame:", len(df))

df['text'] = df['question'] + "\n\n" + df['answer']
hf_dataset = Dataset.from_pandas(df[['text']])
print("Dataset Schema:", hf_dataset)
print("Sample Data:", hf_dataset[0])

hf_dataset = hf_dataset.filter(lambda example: example["text"] is not None and isinstance(example["text"], str))

def tokenize_function(examples):
    texts = examples["text"]
    if isinstance(texts, list):
        texts = [str(text) for text in texts]
    else:
        texts = [str(texts)]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(hf_dataset[0])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./llama-retrained",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none",
    fp16=True,
    dataloader_num_workers=2,
    push_to_hub=False,
    hub_model_id="vipasai/llama-retrained-disease-dataset",
    hub_token=huggingface_token,
    no_cuda=False
)

train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./llama-retrained")
tokenizer.save_pretrained("./llama-retrained")

trainer.push_to_hub(commit_message="Initial retrained LLaMA model upload")
tokenizer.push_to_hub("vipasai/llama-retrained-disease-dataset", commit_message="Add tokenizer")
model.config.push_to_hub("vipasai/llama-retrained-disease-dataset", commit_message="Add model config")

print("Model retraining completed and uploaded to the Hugging Face Hub!")
