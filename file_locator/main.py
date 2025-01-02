# Standard library imports
import json
import os
import random
import subprocess

# Third-party imports
import jsonlines
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import RobertaModel
from typing import List, Dict

# Local application imports
from dependency_analyzer import cal_dep_score
from dependency_analyzer import DependencyClassifier
from siamese_net import evaluate_embedding_model
from siamese_net import load_siamese_data
from siamese_net import train_embedding_model


def mv_large_file(dataset: List[Dict]) -> List[Dict]:
    """
    This function filters out large files from the dataset.
    """
    new_dataset = []
    for sample in dataset:
        if len(sample['file'].splitlines()) // 30 + 1 <= 25:
            new_dataset.append(sample)
    return new_dataset


def list_files_in_directory(git_repo_path: str, sha: str, user_name: str) -> list[str]:
    """
    This function lists files in a directory for a given git repository and commit SHA.
    """
    if os.path.exists(git_repo_path) == False:
        repos_path, proj_name = git_repo_path.rsplit('/', 1)
        print(f'Cloning {proj_name} to {repos_path}')
        # git clone the repo to the path
        command = f'git clone https://github.com/{user_name}/{proj_name}.git {git_repo_path}'
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error executing command: {e}')
            print(f'Error output: {e.output}')
            return None
    command = f'git -C {git_repo_path} ls-tree --name-only -r {sha}'
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        )
        file_list = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f'Error executing command: {e}')
        print(f'Error output: {e.output}')
        return None
    return file_list


def main(
    lang: str,
    recalculate_dep_score: bool,
    test_only: bool,
    debug_mode: bool,
    dataset_root: str = './dataset',
    model_root: str = './model',
    result_root: str = './result',
    device: torch.device = torch.device('cuda', index=0),
) -> None:
    # Step 1: make dataset
    dataset_dir = os.path.join(dataset_root, lang)
    os.makedirs(dataset_dir, exist_ok=True)
    for mode in ['train', 'dev', 'test']:
        jsonl_path = os.path.join(dataset_dir, f'{mode}.jsonl')
        print(f'Load dataset from {jsonl_path}')
        with jsonlines.open(jsonl_path, 'r') as reader:
            dataset = [obj for obj in reader]

        # Step 1: go through dependency analyzer first
        if 'dependency_score' not in dataset[0].keys() or recalculate_dep_score:
            dependency_analyzer = DependencyClassifier(device=device)

            for sample in tqdm(dataset, desc='Calculating dependency score'):
                hunk = sample['hunk']
                file_content = sample['file']
                dep_score_list = cal_dep_score(
                    hunk, file_content, dependency_analyzer)
                sample['dependency_score'] = dep_score_list

            with jsonlines.open(jsonl_path, mode='w') as writer:
                writer.write_all(dataset)

    # Step 2: load datasets
    with open(os.path.join(dataset_dir, 'train.jsonl'), encoding='utf-8') as f:
        train_dataset = [json.loads(line) for line in f.readlines()]
    with open(os.path.join(dataset_dir, 'dev.jsonl'), encoding='utf-8') as f:
        val_dataset = [json.loads(line) for line in f.readlines()]
    with open(os.path.join(dataset_dir, 'test.jsonl'), encoding='utf-8') as f:
        test_dataset = [json.loads(line) for line in f.readlines()]

    # Step 3: Train a siamese network to learn embeddings
    embedding_model = RobertaModel.from_pretrained(
        'huggingface/CodeBERTa-small-v1')
    tokenizer = RobertaTokenizer.from_pretrained(
        'huggingface/CodeBERTa-small-v1')
    if not test_only:
        tensor_train_dataset = load_siamese_data(
            train_dataset, tokenizer, debug_mode)
        tensor_val_dataset = load_siamese_data(
            val_dataset, tokenizer, debug_mode)
        train_dataloader = DataLoader(
            tensor_train_dataset, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(
            tensor_val_dataset, batch_size=1, shuffle=True)
        epoch = 1 if debug_mode else 4
        train_embedding_model(
            embedding_model, train_dataloader, val_dataloader, 1e-5, epoch, lang
        )
    else:
        model_path = os.path.join(model_root, lang, 'embedding_model.bin')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'{model_path} not found')
        embedding_model.load_state_dict(
            torch.load(model_path, map_location=device))

    # Step 4: Calculate the semantic similarity between the edit and the file
    # for val & test dataset
    tensor_val_dataset = load_siamese_data(val_dataset, tokenizer, debug_mode)
    tensor_test_dataset = load_siamese_data(
        test_dataset, tokenizer, debug_mode)
    val_dataloader = DataLoader(
        tensor_val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(
        tensor_test_dataset, batch_size=1, shuffle=False)
    val_embedding_similiarity = evaluate_embedding_model(
        embedding_model, val_dataloader, 'valid'
    )
    test_embedding_similiarity = evaluate_embedding_model(
        embedding_model, test_dataloader, 'test'
    )

    # Step 5: Use linear regression to fit val dataset, and evaluate on test
    # dataset
    X_train = [
        val_dataset[idx]['dependency_score'][:1] +
        [val_embedding_similiarity[idx]]
        for idx in range(len(val_embedding_similiarity))
    ]
    y_train = [val_dataset[idx]['label']
               for idx in range(len(val_embedding_similiarity))]
    X_test = [
        test_dataset[idx]['dependency_score'][:1] +
        [test_embedding_similiarity[idx]]
        for idx in range(len(test_embedding_similiarity))
    ]
    y_test = [test_dataset[idx]['label']
              for idx in range(len(test_embedding_similiarity))]

    reg = LinearRegression().fit(X_train, y_train)
    print(f'Coefficient: {reg.coef_}')
    print(f'Intercept: {reg.intercept_}')
    y_pred = reg.predict(X_test)

    # save y_pred and y_test for further analysis
    os.makedirs(result_root, exist_ok=True)
    with open(os.path.join(result_root, f'{lang}_val.json'), 'w') as f:
        json.dump({'X_val': np.array(X_train).tolist(),
                  'y_val': y_train}, f, indent=4)
    with open(os.path.join(result_root, f'{lang}_test.json'), 'w') as f:
        json.dump({'X_test': np.array(X_test).tolist(),
                  'y_test': y_test}, f, indent=4)
    threshold = {
        'java': 0.4,
        'javascript': 0.4,
        'python': 0.4,
        'go': 0.4,
        'typescript': 0.4,
    }
    y_pred = [1 if y > threshold[lang] else 0 for y in y_pred]

    # Step 7: calculate accuracy, precision, recall, f1
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(
        f'Lang: {lang}, Linear regression model result (combine with dependency score):')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1: {f1 * 100:.2f}%')


if __name__ == '__main__':
    random.seed(42)
    lang = 'javascript'
    recalculate_dep_score = False  # if true, calculate dependency score again
    test_only = True  # if true, only test
    debug_mode = False
    dataset_root: str = './dataset'
    model_root: str = './model'
    result_root: str = './result'
    device = torch.device('cuda', index=0)
    main(
        lang,
        recalculate_dep_score,
        test_only,
        debug_mode,
        dataset_root=dataset_root,
        model_root=model_root,
        result_root=result_root,
        device=device,
    )
