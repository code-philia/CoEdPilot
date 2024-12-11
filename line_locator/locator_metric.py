import bleu
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

def all_in_one(output, gold):
    with open(output, 'r') as f:
        predictions = f.readlines()
    with open(gold, 'r') as f:
        ground_truth = f.readlines()

    # same line number:
    assert len(predictions) == len(ground_truth), "The length of predictions and ground truth must be the same."

    em = 0
    total_label = 0
    match_label = 0
    for i in range(len(predictions)):
        if predictions[i].strip() == ground_truth[i].strip():
            em += 1
        pd = predictions[i].split('\t')[-1].strip().split()
        gt = ground_truth[i].split('\t')[-1].strip().split()
        if len(pd) == len(gt):
            for j in range(len(pd)):
                total_label += 1
                if pd[j] == gt[j]:
                    match_label += 1

    print(f'EM: {em / len(predictions) * 100:.2f}%')
    print(f'Accuracy: {match_label / total_label * 100:.2f}%')

    y_pred = []
    y_true = []
    label_map = {'keep': 0, 'add': 1, 'replace': 2}
    for i in range(len(predictions)):
        pd = predictions[i].split('\t')[-1].strip().split()
        gt = ground_truth[i].split('\t')[-1].strip().split()
        if len(pd) == len(gt):
            y_pred.extend([label_map[x] for x in pd])
            y_true.extend([label_map[x] for x in gt])

    # 计算多标签的精确度
    precision = precision_score(y_true, y_pred, average='macro')
    print(f'Precision(macro): {precision * 100:.2f}%')

    # 计算多标签的召回率
    recall = recall_score(y_true, y_pred, average='macro')
    print(f'Recall(macro): {recall * 100:.2f}%')

    # 计算多标签的F1分数
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'F1(macro): {f1 * 100:.2f}%')


if __name__ == '__main__':
    lang = 'go'
    output_path = f'./model/{lang}/test_1.output'
    gold_path = f'./model/{lang}/test_1.gold'
    all_in_one(output_path, gold_path)