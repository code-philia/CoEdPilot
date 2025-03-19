# README
## Content
* `bleu.py`: calculate the bleu score;
* `generator_metric.py`: calculate the performance (BLEU & Exact match) of generator;
* `model.py`: Seq2Seq model and beam search;
* `run.py`: training, validation and test script;
* `evaluate.sh`: bash script for evaluation;
* `run.sh`: bash script for train and validation;
* `model/`: folder save models and outputs by language, available for download at [Huggingface](https://huggingface.co/code-philia/CoEdPilot-generator);
    * model/{lang}/test_0_pred_gold.json: for each sample, have 10 prediction candidate and 1 ground-truth
    * model/{lang}/checkpoint-best-bleu/pytorch_model.bin: trained model
* `dataset/`: folder save datasets by langauge, available for download at [Huggingface](https://huggingface.co/datasets/code-philia/CoEdPilot-generator).
    * dataset/{lang}/train.jsonl: training dataset
    * dataset/{lang}/dev.jsonl: validation dataset
    * dataset/{lang}/test.jsonl: test dataset

## Download
Download model and dataset by runing download script:
```shell
bash download.sh
```

## Quick start 
Please refer to `Quick_start.ipynb` for quick start.

## As baseline
1. Parpare `custon_input.jsonl` file as input, each element of format:
    > ⚠️: The `edit_labels` and `code_window` should have the same length

    ```python
    {
        "edit_labels": list["replace" | "keep" | "add"]. Each element is an edit operation label,
        "code_window": list[str]. Each str element is a line of code,
        "after_edit": list[str]. If code window contains code block to replace then after_edit is code to replace that block. If code window require add action then after_edit is code to insert.
        "prompt": str | None. The natural language description of the edit,
        "prior_edits": list[dict]. Each dict has the following format: 
            {
                "code_before": list[str]. Each str element is a line of code before editing,
                "code_after":  list[str]. Each str element is a line of code after editing
            }
    }
    ```
2. Run script `transform.py` to transform your customized dataset into CoEdPilot prompt format,  where the three arguments represent the ratio for splitting your dataset into the training set, validation set, and test set.
    ```python
    python transform.py 7 2 1
    ```
3. You can now run or evaluate on your customized dataset 

## Usage
```shell
# Train
bash run.sh

# Evaluate
bash evaluate.sh
```

## Key results
|  Language  |  EM@1 |  EM@3 |  EM@5 | EM@10 | BLEU@1| BLEU@3| BLEU@5| BLEU@10|
|------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Typescript | 41.58 | 46.86 | 48.75 | 50.65 | 61.75 | 70.31 | 71.99 | 73.68 |
| Javascript | 41.83 | 47.50 | 49.31 | 50.99 | 60.70 | 69.71 | 71.37 | 73.02 |
| Java       | 40.69 | 46.87 | 48.78 | 50.51 | 60.54 | 68.35 | 70.11 | 71.73 | 
| Python     | 33.48 | 38.52 | 40.41 | 42.09 | 57.59 | 65.65 | 67.47 | 69.11 |
| Go         | 48.94 | 55.09 | 57.18 | 59.16 | 65.37 | 71.96 | 73.47 | 74.98 |
