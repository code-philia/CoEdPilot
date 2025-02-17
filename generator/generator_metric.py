import json

import bleu


def all_in_one(output):
    with open(output, 'r') as f:
        test_set = json.load(f)

    print('Total:', len(test_set))
    em_1 = 0
    em_3 = 0
    em_5 = 0
    em_10 = 0
    for key, sample in test_set.items():
        predictions, ground_truth = sample[0], sample[1]
        for idx, prediction in enumerate(predictions):
            if prediction.split('\t')[-1].strip() == ground_truth.split('\t')[-1].strip():
                if idx < 1:
                    em_1 += 1
                    em_3 += 1
                    em_5 += 1
                    em_10 += 1
                    break
                elif idx < 3:
                    em_3 += 1
                    em_5 += 1
                    em_10 += 1
                    break
                elif idx < 5:
                    em_5 += 1
                    em_10 += 1
                    break
                elif idx < 10:
                    em_10 += 1
                    break

    print('EM@1:', round(em_1 / len(test_set) * 100, 2))
    print('EM@3:', round(em_3 / len(test_set) * 100, 2))
    print('EM@5:', round(em_5 / len(test_set) * 100, 2))
    print('EM@10:', round(em_10 / len(test_set) * 100, 2))

    (goldMap, predictionMap) = bleu.computeMaps_multiple(output_path, 1)
    bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    print('BLEU-4@1:', bleu_score)
    (goldMap, predictionMap) = bleu.computeMaps_multiple(output_path, 3)
    bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    print('BLEU-4@3:', bleu_score)
    (goldMap, predictionMap) = bleu.computeMaps_multiple(output_path, 5)
    bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    print('BLEU-4@5:', bleu_score)
    (goldMap, predictionMap) = bleu.computeMaps_multiple(output_path, 10)
    bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    print('BLEU-4@10:', bleu_score)


if __name__ == '__main__':
    output_path = './model/go/test_0_pred_gold.json'
    all_in_one(output_path)
