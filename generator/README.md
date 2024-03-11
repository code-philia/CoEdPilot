# README
## Files:
* bleu.py: calculate the bleu score
* generator_metric.py: calculate the performance (BLEU & Exact match) of generator
* model.py: Seq2Seq model and beam search
* run.py: training, validation and test script
* evaluate.sh: bash script for evaluation
* run.sh: bash script for train and validation
* model/: folder save models and outputs by language
    * model/{lang}/test_0_pred_gold.json: for each sample, have 10 prediction candidate and 1 ground-truth
    * model/{lang}/checkpoint-best-bleu/pytorch_model.bin: trained model
* dataset/: folder save datasets by langauge

## Generator Performance
|  Language  |  EM@1 |  EM@3 |  EM@5 | EM@10 | BLEU@1| BLEU@3| BLEU@5| BLEU@10|
|------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Typescript | 41.58 | 46.86 | 48.75 | 50.65 | 61.75 | 70.31 | 71.99 | 73.68 |
| Javascript | 41.83 | 47.50 | 49.31 | 50.99 | 60.70 | 69.71 | 71.37 | 73.02 |
| Java       | 40.69 | 46.87 | 48.78 | 50.51 | 60.54 | 68.35 | 70.11 | 71.73 | 
| Python     | 33.48 | 38.52 | 40.41 | 42.09 | 57.59 | 65.65 | 67.47 | 69.11 |
| Go         | 48.94 | 55.09 | 57.18 | 59.16 | 65.37 | 71.96 | 73.47 | 74.98 |
