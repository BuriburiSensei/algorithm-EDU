import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


base_model_name = "Qwen/Qwen2-7B"
lora_path = "./lora-medical"


tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)


model = PeftModel.from_pretrained(model, lora_path)

model.eval()


def build_prompt(user_input):
    system_prompt = "你是一名临床经验丰富的医生，请基于循证医学进行逐步推理，并给出诊断依据和建议。"

    prompt = f"""System: {system_prompt}
User: {user_input}
Assistant:"""

    return prompt


def generate_answer(user_input):
    prompt = build_prompt(user_input)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Assistant:" in result:
        result = result.split("Assistant:")[-1].strip()

    return result


if __name__ == "__main__":
    while True:
        user_input = input("请输入患者情况（输入exit退出）：\n> ")

        if user_input.lower() == "exit":
            break

        answer = generate_answer(user_input)

        print("\n模型回答：\n")
        print(answer)
        print("\n" + "="*50 + "\n")