from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

def all_in_one(output, gold):
    with open(output, 'r') as f:
        predictions = f.readlines()
    with open(gold, 'r') as f:
        ground_truth = f.readlines()

    # same line number:
    assert len(predictions) == len(
        ground_truth), "The length of predictions and ground truth must be the same."

    em = 0
    total_label = 0
    match_label = 0
    for pd, gt in zip(predictions, ground_truth):
        if pd.strip() == gt.strip():
            em += 1
        pd = pd.split('\t')[-1].strip().split()
        gt = gt.split('\t')[-1].strip().split()
        if len(pd) == len(gt):
            for j in range(len(pd)):
                total_label += 1
                if pd[j] == gt[j]:
                    match_label += 1

    em = em / len(predictions)
    acc = match_label / total_label
    return em, acc

def calc_precision_recall_f1(predictions: List[str], ground_truth: List[str]) -> Tuple[float, float, float]:
    """Calculate precision, recall and f1 score between predictions and ground truth."""
    y_pred = []
    y_true = []
    label_map = {'keep': 0, 'add': 1, 'replace': 2}
    for pd, gt in zip(predictions, ground_truth):
        pd = pd.split('\t')[-1].strip().split()
        gt = gt.split('\t')[-1].strip().split()
        if len(pd) == len(gt):
            y_pred.extend([label_map[x] for x in pd])
            y_true.extend([label_map[x] for x in gt])

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

def all_in_one(output: str, gold: str) -> None:
    # load from files
    with open(output, 'r') as f:
        predictions = f.readlines()
    with open(gold, 'r') as f:
        ground_truth = f.readlines()

    # same line number:
    assert len(predictions) == len(ground_truth), "The length of predictions and ground truth must be the same."

    em, acc = calc_em_acc(predictions, ground_truth)
    print(f'EM: {em * 100:.2f}%')
    print(f'Accuracy: {acc * 100:.2f}%')

    precision, recall, f1 = calc_precision_recall_f1(predictions, ground_truth)
    print(f'Precision(macro): {precision * 100:.2f}%')
    print(f'Recall(macro): {recall * 100:.2f}%')
    print(f'F1(macro): {f1 * 100:.2f}%')


if __name__ == '__main__':
    lang = 'go'
    output_path = f'./model/{lang}/test_1.output'
    gold_path = f'./model/{lang}/test_1.gold'
    all_in_one(output_path, gold_path)
