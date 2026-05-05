# LLM Hallucinations — RLVR (Practical)

Практическая часть диплома по снижению галлюцинаций в LLM через RLVR‑прототип и контрольный RAG.

## Старт в Colab

```bash
!git clone https://github.com/dependonsincerity/llm-hallucinations-rlvr.git
%cd llm-hallucinations-rlvr
!pip install -r requirements.txt

Быстрый baseline

python scripts/download_data.py
python scripts/run_baseline.py --model qwen
python scripts/run_baseline.py --model saiga
python scripts/eval_metrics.py --run_id qwen_baseline_outputs
python scripts/eval_metrics.py --run_id saiga_baseline_outputs

Структура

data/ — данные
scripts/ — запуск и оценка
results/ — результаты
configs/ — конфиги
