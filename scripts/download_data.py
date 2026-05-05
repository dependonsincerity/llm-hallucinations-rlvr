import os
import pandas as pd

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def build_ru_qa_mini(path):
    data = [
        {"question": "Столица Франции?", "answer": "Париж"},
        {"question": "Столица Германии?", "answer": "Берлин"},
        {"question": "Столица Италии?", "answer": "Рим"},
        {"question": "Столица Испании?", "answer": "Мадрид"},
        {"question": "Сколько будет 7 + 8?", "answer": "15"},
        {"question": "Сколько будет 12 * 3?", "answer": "36"},
        {"question": "Кто написал «Войну и мир»?", "answer": "Лев Толстой"},
        {"question": "Кто написал «Преступление и наказание»?", "answer": "Фёдор Достоевский"},
        {"question": "Самая большая планета Солнечной системы?", "answer": "Юпитер"},
        {"question": "Самая близкая к Солнцу планета?", "answer": "Меркурий"},
        {"question": "Самая большая океаническая впадина?", "answer": "Марианская впадина"},
        {"question": "Сколько континентов на Земле?", "answer": "6"},
        {"question": "Какая река самая длинная в мире?", "answer": "Нил"},
        {"question": "Самая высокая гора мира?", "answer": "Эверест"},
        {"question": "Столица России?", "answer": "Москва"},
        {"question": "Столица Японии?", "answer": "Токио"},
        {"question": "Самая высокая гора России?", "answer": "Эльбрус"},
        {"question": "Кто открыл закон всемирного тяготения?", "answer": "Исаак Ньютон"},
        {"question": "Основной газ в атмосфере Земли?", "answer": "Азот"},
        {"question": "Формула воды?", "answer": "H2O"},
    ]
    pd.DataFrame(data).to_csv(path, index=False)

def download_truthfulqa(path):
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "generation")["validation"]
        rows = []
        for ex in ds:
            rows.append({
                "question": ex["question"],
                "answers": "|".join(ex["correct_answers"]),
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        print("TruthfulQA saved:", path)
    except Exception as e:
        print("TruthfulQA download failed:", e)
        print("You can still run RU mini QA.")

if __name__ == "__main__":
    ensure_dirs()
    download_truthfulqa("data/processed/truthfulqa.csv")
    build_ru_qa_mini("data/processed/ru_qa_mini.csv")
    print("Done.")
