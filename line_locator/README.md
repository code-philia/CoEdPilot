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

## Key results
|  Language  |  EM   |  Acc  |  Prec | Recal |   F1  |
|------------|-------|-------|-------|-------|-------|
| Typescript | 76.75 | 95.23 | 86.21 | 84.25 | 85.20 |
| Javascript | 74.62 | 94.89 | 86.62 | 83.88 | 85.18 |
| Java       | 78.00 | 95.37 | 87.99 | 85.99 | 86.96 |
| Python     | 74.44 | 94.48 | 85.03 | 82.64 | 83.79 |
| Go         | 80.18 | 95.79 | 88.99 | 87.32 | 88.14 |

## Usage
### Training
Modify hyperparameters in `run.py` and run the following command:
```shell
bash run.sh
```
### Evaluation
Modify hyperparameters in `evaluate.sh` and run the following command:
```shell
bash evaluate.sh
```