# Standard library imports
import os

# Third-party imports
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaConfig
from transformers import RobertaModel
from transformers import RobertaTokenizer


def train_embedding_model(
    model: RobertaModel,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    lr: float,
    epochs: int,
    lang: str,
) -> None:
    """
    Train the embedding model using the siamese network approach.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()

    for i in range(epochs):
        core_dataset = []
        # 1. embed data samples to find the most similar pair (prior edit, file
        # sliding window)
        pbar = tqdm(
            train_dataloader, desc=f'Find most similar pair for epoch {i+1} / {epochs}'
        )
        with torch.no_grad():
            model.eval()
            for batch in pbar:
                # input_ids: [batch_size(1), 1 + file_line // window_len, max_length]
                # attn_masks: [batch_size(1), 1 + file_line // window_len, max_length]
                _, input_ids, attn_masks, label = [b.to(device) for b in batch]
                input_ids = input_ids.squeeze(0)
                attn_masks = attn_masks.squeeze(0)
                label = label.squeeze(0)
                dataloader_in_batch = DataLoader(
                    list(zip(input_ids, attn_masks)), batch_size=20, shuffle=False
                )
                all_embeddings = []
                for input_ids_in_batch, attn_masks_in_batch in dataloader_in_batch:
                    hidden_states = model(
                        input_ids_in_batch, attn_masks_in_batch
                    ).last_hidden_state
                    embeddings = torch.mean(hidden_states, dim=1)  # Average pooling
                    all_embeddings.append(embeddings)

                embeddings = torch.cat(all_embeddings, dim=0)
                edit_embedding = embeddings[0:1]
                file_embeddings = embeddings[1:]

                # calculate similarity
                similarity = F.cosine_similarity(edit_embedding, file_embeddings, dim=1)

                # find file_embedding with max similarity
                max_similarity_idx = torch.argmax(similarity)

                core_dataset.append(
                    [
                        input_ids[0].detach(),
                        attn_masks[0].detach(),
                        input_ids[max_similarity_idx + 1].detach(),
                        attn_masks[max_similarity_idx + 1].detach(),
                        label.detach(),
                    ]
                )
        torch.cuda.empty_cache()

        # 2. train the model with the most similar pair
        core_dataloader = DataLoader(core_dataset, batch_size=16, shuffle=True)
        pbar = tqdm(core_dataloader, desc=f'Train epoch {i+1} / {epochs}')
        model.train()
        for batch in pbar:
            # edit_input_ids: [batch_size, max_length]
            # edit_attn_masks: [batch_size, max_length]
            # max_similiarity_input_ids: [batch_size, max_length]
            # max_similiarity_attn_masks: [batch_size, max_length]
            # label: [batch_size]
            (
                edit_input_ids,
                edit_attn_masks,
                max_similiarity_input_ids,
                max_similiarity_attn_masks,
                label,
            ) = [b.to(device) for b in batch]

            edit_embedding = model(edit_input_ids, edit_attn_masks)
            edit_embedding = torch.mean(
                edit_embedding.last_hidden_state, dim=1
            )  # Average pooling
            max_similiarity_embedding = model(
                max_similiarity_input_ids, max_similiarity_attn_masks
            )
            max_similiarity_embedding = torch.mean(
                max_similiarity_embedding.last_hidden_state, dim=1
            )  # Average pooling

            # contrastive loss between edit_embedding and
            # max_similiarity_embedding
            loss = criterion(edit_embedding, max_similiarity_embedding, label.squeeze(1))

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # save model
        if not os.path.exists(f'./model/{lang}'):
            os.makedirs(f'./model/{lang}')
        torch.save(model.state_dict(), f'./model/{lang}/embedding_model.bin')

        # evaluate
        evaluate_embedding_model(model, dev_dataloader, 'validation')


def load_siamese_data(
    dataset: list[dict], tokenizer: RobertaTokenizer, debug_mode: bool = False
) -> list:
    def split2window(lines: list, window_len: int = 30) -> list:
        windows = []
        for i in range(len(lines) // window_len + 1):
            if i == len(lines) // window_len:
                window = "".join(lines[i * window_len :])
            else:
                window = "".join(lines[i * window_len : (i + 1) * window_len])
            windows.append(window)
        return windows

    tensor_dataset = []
    for sample_idx, sample in enumerate(tqdm(dataset, desc='Loading data')):
        hunk = sample['hunk']
        file = sample['file']
        file_windows = split2window(file.splitlines(True))
        input = ["".join(hunk['code_window'])] + file_windows
        tensor_input = tokenizer(
            input, return_tensors='pt', padding='max_length', truncation=True, max_length=512,
        )
        if sample['label'] == 0:
            label = -1
        else:
            label = 1
        tensor_dataset.append(
            (
                torch.tensor([sample['dependency_score'][0]]),
                tensor_input['input_ids'],
                tensor_input['attention_mask'],
                torch.tensor([label], dtype=torch.float),
            )
        )
        if debug_mode:
            if sample_idx >= 100:
                break

    return tensor_dataset


def evaluate_embedding_model(
    model: RobertaModel, dataloader: DataLoader, mode: str
) -> np.array:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    preds = []
    golds = []
    for batch in tqdm(dataloader, desc='Evaluating'):
        _, input_ids, attn_masks, label = [b.to(device) for b in batch]
        input_ids = input_ids.squeeze(0)
        attn_masks = attn_masks.squeeze(0)

        dataloader_in_batch = DataLoader(
            list(zip(input_ids, attn_masks)), batch_size=16, shuffle=False
        )
        all_embeddings = []
        with torch.no_grad():
            for input_ids_in_batch, attn_masks_in_batch in dataloader_in_batch:
                embeddings = model(input_ids_in_batch, attn_masks_in_batch).last_hidden_state[
                    :, 0, :
                ]
                all_embeddings.append(embeddings)
            all_embeddings = torch.cat(all_embeddings, dim=0)

        edit_embedding = all_embeddings[0:1]
        file_embeddings = all_embeddings[1:]

        # calculate similarity
        similarity = F.cosine_similarity(edit_embedding, file_embeddings, dim=1)

        # get max similiarity as prediction
        preds.append(torch.max(similarity).detach().cpu().numpy())
        golds.append(label.squeeze(0).detach().cpu().numpy())

    preds = np.array(preds)
    golds = np.array(golds)

    return preds
