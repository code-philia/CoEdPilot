# README
## Content
* `dependency_analyzer.py`: load dependency analyzer and estimate dependency score;
* `siamise_model.py`: load, train and inference semantic similarity model;
* `main.py`: train semantic similarity model and fit a linear regression model to estimate edit propagation score of a file;
* `dataset/`: dataset for training and evaluation, available in [Huggingface](https://huggingface.co/datasets/code-philia/CoEdPilot-file-locator);
* `models/`: semantic embedding models, available in [Huggingface](https://huggingface.co/code-philia/CoEdPilot-file-locator).

## Key results
| Language    | Accuracy | Precision | Recall | F1     |
|-------------|----------|-----------|--------|--------|
| JavaScript  | 88.77%   | 81.52%    | 71.21% | 76.02% |
| Python      | 85.78%   | 70.84%    | 73.40% | 72.09% |
| Java        | 90.65%   | 85.28%    | 75.67% | 80.19% |
| Go          | 88.55%   | 80.10%    | 72.12% | 75.90% |
| Typescript  | 88.46%   | 79.84%    | 72.25% | 75.86% |
| Average     | 88.44%   | 79.52%    | 72.93% | 76.01% |

## Usage
* Adjust hyperparameters in `main.py` and run `python main.py` to train semantic similarity model and fit a linear regression model to estimate edit propagation score of a file.
