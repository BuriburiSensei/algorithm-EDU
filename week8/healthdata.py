import json
import time
import openai
from tqdm import tqdm
import random


# 1. 配置

openai.api_key = ""
openai.api_base = ""

MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"

OUTPUT_PATH = "medical_cot_500.jsonl"

TARGET_SIZE = 50
MAX_RETRY = 3
SLEEP_TIME = 1



# 2. 随机医疗场景

SYMPTOMS_POOL = [
    "发烧、咳嗽", "头痛、恶心", "胸痛、呼吸困难",
    "腹痛、腹泻", "皮疹、瘙痒", "关节疼痛",
    "乏力、体重下降", "心悸、出汗", "视力模糊"
]

PATIENT_TYPE = [
    "一名30岁男性", "一名25岁女性", "一名老年人",
    "一名儿童", "一名中年患者"
]


def build_prompt():
    symptom = random.choice(SYMPTOMS_POOL)
    patient = random.choice(PATIENT_TYPE)

    return f"""
你是一个经验丰富的医生，请生成一个医疗问答数据，并包含详细推理过程（Chain-of-Thought）。

要求：
1. 构造一个真实合理的医疗问题（包含患者信息和症状）
2. 给出逐步推理（Step 1, Step 2...）
3. 推理必须医学合理（不能胡编）
4. 可以给出可能疾病和建议检查
5. 最后给出 Conclusion（最终判断）

输出格式：

Question: ...
Step 1: ...
Step 2: ...
...
Conclusion: ...
"""


# 3. 调用LLM

def call_llm(prompt):
    for _ in range(MAX_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )

        
            if response is None:
                print(" response is None")
                continue

            if not isinstance(response, dict):
                print(" response not dict:", response)
                continue

            choices = response.get("choices", None)

            if choices is None:
                print(" choices is None:", response)
                continue

            if not isinstance(choices, list):
                print(" choices not list:", choices)
                continue

            if len(choices) == 0:
                print(" empty choices")
                continue

            message = choices[0].get("message", None)

            if message is None:
                print(" message missing:", choices[0])
                continue

            content = message.get("content", None)

            if content is None:
                print(" content missing:", message)
                continue

            return content

        except Exception as e:
            print("Retrying due to error:", e)
            time.sleep(2)

    return None



# 4. 解析输出

def parse_output(text):
    if text is None:
        return None

    if "Question:" not in text:
        return None

    try:
        q_part = text.split("Question:")[1]
        question, rest = q_part.split("Step 1:", 1)

        cot = "Step 1:" + rest

        if "Conclusion:" in cot:
            cot_part, answer = cot.split("Conclusion:", 1)
        else:
            cot_part = cot
            answer = ""

        return {
            "instruction": question.strip(),
            "cot": cot_part.strip(),
            "answer": answer.strip()
        }
    except:
        return None



# 5. 过滤

def quality_filter(sample):
    if sample is None:
        return False

    cot = sample["cot"]

    if "Step 1" not in cot:
        return False
    if len(cot) < 80:
        return False
    if "Conclusion" not in cot and len(sample["answer"]) == 0:
        return False

    return True





def main():
    results = []
    seen_questions = set()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        pbar = tqdm(total=TARGET_SIZE)

        while len(results) < TARGET_SIZE:

            prompt = build_prompt()
            text = call_llm(prompt)

            if text is None:
                print(" skip due to LLM failure")
                continue

            sample = parse_output(text)

            if sample is None:
                print(" parse failed")
                continue


            if not quality_filter(sample):
                continue

            if sample["instruction"] in seen_questions:
                continue

            seen_questions.add(sample["instruction"])
            results.append(sample)

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fout.flush()

            pbar.update(1)

            time.sleep(SLEEP_TIME)

        pbar.close()


if __name__ == "__main__":
    main()