# README
## Contents
* `bleu.py`: calculate the bleu score;
* `locator_metric.py`: calculate the performance (EM, Acc, Prec, Recall, F1) of locator;
* `model.py`: masked language modelling model;
* `run.py`: training, validation and test script;
* `evaluate.sh`: bash script for evaluation;
* `run.sh`: bash script for train and validation;
* `model/`: folder save models and outputs by language, available for download at [Huggingface](https://huggingface.co/code-philia/CoEdPilot-line-locator);
    * model/{lang}/test_1.gold: ground truth
    * model/{lang}/test_1.output: prediction
    * model/{lang}/checkpoint-best-bleu/pytorch_model.bin: trained model
* `dataset/`: folder save datasets by langauge, available for download at [Huggingface](https://huggingface.co/datasets/code-philia/CoEdPilot-line-locator).
    * dataset/{lang}/train.jsonl: training dataset
    * dataset/{lang}/dev.jsonl: validation dataset
    * dataset/{lang}/test.jsonl: test dataset

## Download
Download model and dataset by runing download script:
```shell
bash download.sh
```

## Usage
```shell
# Train
bash run.sh
# Evaluate
bash evaluate.sh
```

## Quick start 
Please refer to `Quick_start.ipynb` for quick start.

## As baseline
1. Prepare `custom_input.jsonl` file as input, each element of format:
    > [!NOTE]
    >
    > The `edit_labels` and `code_window` should have the same length
    ```python
    {
        "edit_labels": list["replace" | "keep" | "add"]. Each element is an edit operation label,
        "code_window": list[str]. Each str element is a line of code,
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

## Key results
|  Language  |  EM   |  Acc  |  Prec | Recal |   F1  |
|------------|-------|-------|-------|-------|-------|
| Typescript | 76.75 | 95.23 | 86.21 | 84.25 | 85.20 |
| Javascript | 74.62 | 94.89 | 86.62 | 83.88 | 85.18 |
| Java       | 78.00 | 95.37 | 87.99 | 85.99 | 86.96 |
| Python     | 74.44 | 94.48 | 85.03 | 82.64 | 83.79 |
| Go         | 80.18 | 95.79 | 88.99 | 87.32 | 88.14 |