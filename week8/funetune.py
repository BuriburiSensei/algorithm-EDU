import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments


# 1. 模型

model_name = "Qwen/Qwen2-7B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.gradient_checkpointing_enable()


# 2. LoRA配置

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)
model = get_peft_model(model, lora_config)


# 3. 加载数据
dataset = load_dataset("json", data_files="medical_cot_500.jsonl")


def format_example(example):
    instruction = example.get("instruction", "")
    cot = example.get("cot", "")
    answer = example.get("answer", "")


    system_prompt = "你是一个专业医生，请基于医学知识进行严谨推理并回答。"

    text = f"""System: {system_prompt}
User: {instruction}
Assistant: {cot}
{answer}"""

    return {"text": text}

dataset = dataset["train"].map(format_example)


# 4. 训练

training_args = TrainingArguments(
    output_dir="./lora-medical",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    report_to="none"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=2048,
    args=training_args
)

trainer.train()

trainer.model.save_pretrained("./lora-medical")
tokenizer.save_pretrained("./lora-medical")

print("training completed")