import os
import json
import argparse


def main(args):
    with open('custom_input.jsonl', 'r') as f:
        dataset = [json.loads(line) for line in f.readlines()]

    transformed_dataset = []
    for sample in dataset:
        docstring_tokens = ' '.join(sample['edit_labels'])
        code_tokens = '<mask> ' + '<mask> '.join(sample['code_window'])
        code_tokens += '</s>' + sample['prompt']
        for edit in sample['prior_edits']:
            if edit['code_before'] != []:
                code_tokens += '</s> remove ' + "".join(edit['code_before'])
            if edit['code_after'] != []:
                code_tokens += '</s> add ' + "".join(edit['code_after'])
        code_tokens += '</s>'
        transformed_dataset.append(
            {'docstring_tokens': docstring_tokens, 'code_tokens': code_tokens}
        )

    train_dataset_to_idx = int(len(transformed_dataset) * args.train_ratio)
    val_dataset_to_idx = train_dataset_to_idx + int(len(transformed_dataset) * args.val_ratio)
    test_dataset_to_idx = val_dataset_to_idx + int(len(transformed_dataset) * args.test_ratio)

    os.makedirs('dataset', exist_ok=True)
    with open('dataset/train.jsonl', 'w') as f:
        for sample in transformed_dataset[:train_dataset_to_idx]:
            f.write(json.dumps(sample) + '\n')

    with open('dataset/val.jsonl', 'w') as f:
        for sample in transformed_dataset[train_dataset_to_idx:val_dataset_to_idx]:
            f.write(json.dumps(sample) + '\n')

    with open('dataset/test.jsonl', 'w') as f:
        for sample in transformed_dataset[val_dataset_to_idx:test_dataset_to_idx]:
            f.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transform dataset into CoEdPilot prompt format and split into training, validation, and test sets.'
    )
    parser.add_argument('train_ratio', type=int, help='Ratio of the training set')
    parser.add_argument('val_ratio', type=int, help='Ratio of the validation set')
    parser.add_argument('test_ratio', type=int, help='Ratio of the test set')

    args = parser.parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio
    args.train_ratio = args.train_ratio / total
    args.val_ratio = args.val_ratio / total
    args.test_ratio = args.test_ratio / total

    print(
        f'Dataset will be split as follows: Training: {args.train_ratio:.4f}%, Validation: {args.val_ratio:.4f}%, Test: {args.test_ratio:.4f}%'
    )
    main(args)
    print('Dataset transformed and split successfully!')
