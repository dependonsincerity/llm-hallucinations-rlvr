import argparse
import yaml
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_config():
    with open("configs/models.yaml", "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    with open("configs/eval.yaml", "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)
    return models_cfg, eval_cfg

def load_model(model_name, load_in_4bit=False, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"device_map": device_map}
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tok, model

def build_prompt(tokenizer, question, lang="en"):
    system = "You are a helpful assistant. Answer concisely and factually."
    if lang == "ru":
        system = "Вы — полезный помощник. Отвечайте кратко и фактически."
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return system + "\nQ: " + question + "\nA:"

def generate_answer(tokenizer, model, prompt, gen_cfg):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            do_sample=gen_cfg["do_sample"],
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def run_for_csv(model_key):
    models_cfg, eval_cfg = load_config()
    cfg = models_cfg[model_key]
    tok, model = load_model(cfg["name"], cfg["load_in_4bit"], cfg["device_map"])

    runs = []
    tqa_path = eval_cfg["paths"]["truthfulqa"]
    try:
        tqa = pd.read_csv(tqa_path)
        for _, row in tqdm(tqa.iterrows(), total=len(tqa), desc="TruthfulQA"):
            prompt = build_prompt(tok, row["question"], lang="en")
            ans = generate_answer(tok, model, prompt, eval_cfg["generation"])
            runs.append({"dataset": "truthfulqa", "question": row["question"], "answer": ans, "gold": row["answers"]})
    except Exception as e:
        print("TruthfulQA skipped:", e)

    ru_path = eval_cfg["paths"]["ru_qa"]
    ru = pd.read_csv(ru_path)
    for _, row in tqdm(ru.iterrows(), total=len(ru), desc="RU-QA"):
        prompt = build_prompt(tok, row["question"], lang="ru")
        ans = generate_answer(tok, model, prompt, eval_cfg["generation"])
        runs.append({"dataset": "ru_qa", "question": row["question"], "answer": ans, "gold": row["answer"]})

    out_path = f"results/logs/{model_key}_baseline_outputs.csv"
    pd.DataFrame(runs).to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "saiga"], required=True)
    args = parser.parse_args()
    run_for_csv(args.model)
