# Cultural Metric Pipeline

This is a RAG-based metric that automatically evaluates how well image generation results reflect the culture of a specific country. It builds a knowledge base from PDFs collected from Wikipedia, etc., then combines question generation LLM and open-source VLM to create yes/no verification questions and calculate scores based on the answers.

By default, it creates 6-8 questions for one image, and at least 2 of them are forced to check for "elements that should not appear" to increase discrimination.

## Directory Structure

```
evaluation/cultural_metric/
├── build_cultural_index.py      # PDF to FAISS knowledge base conversion script
├── cultural_metric_pipeline.py  # RAG + VLM evaluation pipeline
├── legacy/                      # Previous experiment scripts and results storage
├── requirements.txt             # Dependencies list
├── README.md
└── vector_store/                # (Generated) FAISS index + metadata
```

## 1. Install Dependencies

```bash
cd /Users/chan/Downloads/iaseai26/evaluation/cultural_metric
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> When running on a GPU server, install the appropriate `torch` build for your environment separately.

## 2. Build Knowledge Base

```bash
python build_cultural_index.py \
  --pdf-dir /Users/chan/Downloads/iaseai26/external_data \
  --out-dir vector_store \
  --model-name sentence-transformers/all-MiniLM-L6-v2
```

`faiss.index`, `metadata.jsonl`, `index_config.json` are created under `vector_store/`.

## 3. Manual Cultural Metric Execution Example

```bash
python cultural_metric_pipeline.py \
  --input-csv /Users/chan/Downloads/iaseai26/evaluation/generated_csv/qwen/img_paths_standard.csv \
  --image-root /Users/chan/Downloads/iaseai26/dataset \
  --summary-csv /Users/chan/Downloads/iaseai26/evaluation/outputs/qwen/cultural_metrics_manual_summary.csv \
  --detail-csv /Users/chan/Downloads/iaseai26/evaluation/outputs/qwen/cultural_metrics_manual_detail.csv \
  --question-model openai/gpt-oss-20b \
  --vlm-model Qwen/Qwen2-VL-7B-Instruct \
  --max-questions 8 \
  --min-questions 6 \
  --min-negative 2 \
  --top-k 8 \
  --load-in-8bit \
  --load-in-4bit
```

Generated results:
- `*_summary.csv`: Accuracy/Precision/Recall/F1 per image.
- `*_detail.csv`: Detailed log including questions, expected answers, VLM responses, and evidence text.

Usually, running `evaluation/run_all_metrics.py` or `./run_evaluation.sh` will automatically call this script and save CSV files under `evaluation/outputs/<model>/`.

## How to Compare with Human Evaluation

1. Merge `cultural_metrics_*_summary.csv` with existing CLIP/Aesthetic/DreamSim/human evaluations.
2. Select the step with the highest/lowest F1 for each prompt and calculate the match rate with human Best/Worst.
3. Organize statistics such as Spearman correlation, Top-1 matching, error case analysis, etc.
4. If necessary, find frequently wrong questions in `*_detail.csv` and modify prompts or query templates.

## Notes

- VLM must support `AutoProcessor.apply_chat_template`. For models using separate APIs, replace the `VLMClient` part in `cultural_metric_pipeline.py`.
- If you want to use a different embedding model, change `--model-name` when running `build_cultural_index.py` and keep the generated `index_config.json` together.
- The `legacy/` folder contains integrated scripts for 0926 experiments, analysis scripts, and old result CSVs. Refer to them if needed, and keep them independent of the new pipeline.

Happy evaluating!
