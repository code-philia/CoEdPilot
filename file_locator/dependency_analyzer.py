# Standard library imports
import os

# Third-party imports
import concurrent.futures
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import EncoderDecoderModel
from transformers import PreTrainedModel
from transformers import RobertaTokenizerFast
from typing import Union


class DependencyAnalyzer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, encoder: Union[PreTrainedModel, None] = None,
                 match_tokenizer: Union[RobertaTokenizerFast, None] = None,
                 input_features: int = 768,
                 output_features: int = 2):
        super(DependencyAnalyzer, self).__init__()
        if not encoder:
            encoder: PreTrainedModel = EncoderDecoderModel.from_encoder_decoder_pretrained(
                "microsoft/codebert-base", "microsoft/codebert-base").encoder
        if match_tokenizer:
            encoder.resize_token_embeddings(len(match_tokenizer))
            encoder.config.decoder_start_token_id = match_tokenizer.cls_token_id
            encoder.config.pad_token_id = match_tokenizer.pad_token_id
            encoder.config.eos_token_id = match_tokenizer.sep_token_id
            encoder.config.vocab_size = match_tokenizer.vocab_size
        self.encoder = encoder
        self.dense = nn.Linear(input_features, output_features)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        output_2d = self.dense(pooler_output)
        return output_2d

    

def load_model_and_tokenizer(model_dir: str,
                             directly_load: bool = True,
                             model_with_structure_dir: str | None = None) -> tuple[DependencyAnalyzer, RobertaTokenizerFast]:
    if directly_load:
        # 增强模型和分词器加载时的异常处理
        try:
            tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {model_dir}:{e}")
        if model_with_structure_dir:
            try:
                model = DependencyAnalyzer.from_pretrained(model_with_structure_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {model_with_structure_dir}:{e}")
        else:
            try:
                model = DependencyAnalyzer(match_tokenizer=tokenizer)
                model.load_state_dict(
                    torch.load(
                        os.path.join(
                            model_dir,
                            'pytorch_model.bin'), map_location=torch.device('cpu')))
            except Exception as e:
                raise RuntimeError(f"Failed to load model state from {model_dir}: {e}")
        return model, tokenizer

    model = EncoderDecoderModel.from_pretrained(model_dir)
    if not isinstance(model, EncoderDecoderModel):
        raise RuntimeError(f"Model read from {model_dir} is not valid")
    model = model.encoder
    if not isinstance(model, PreTrainedModel):
        raise RuntimeError(f"Encoder of original model is not valid")

    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(
        "microsoft/codebert-base")
    if not isinstance(tokenizer, RobertaTokenizerFast):
        raise RuntimeError("Cannot read tokenizer as microsoft/codebert-base")
    special_tokens = ['<from>', '<to>']
    # tokenizer.add_tokens(my_tokens, special_tokens = False)
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    model = DependencyAnalyzer(model, tokenizer)

    return model, tokenizer


class DependencyClassifier:
    def __init__(self,
                 load_dir: str = "../dependency_analyzer/model",
                 load_with_model_struture: bool = False,
                 device: torch.device = torch.device("cuda", index=0)):
        self.model, self.tokenizer = load_model_and_tokenizer(load_dir, model_with_structure_dir=load_dir) \
            if load_with_model_struture \
            else load_model_and_tokenizer(load_dir)
        self.device = device
        self.model.to(self.device)

    def construct_pair(self, code_1: str, code_2: str):
        if not isinstance(code_1, str) or not isinstance(code_2, str):
            raise ValueError("Both code_1 and code_2 must be strings.")
        if not code_1 or not code_2:
            raise ValueError("Both code_1 and code_2 must not be empty.")
        return '<from>' + code_1 + '<to>' + code_2

    def construct_corpus_pair(self, corpus: list[tuple[str, str]]):
        return [self.construct_pair(code_1, code_2)
                for code_1, code_2 in corpus]

    def gen(self, text: str):
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        text = text.strip() # 去除首尾空白字符
        # ATTENTION: converted to batch here
        token_input = self.tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            token_input = token_input.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=token_input['input_ids'],
                attention_mask=token_input['attention_mask']
            )[0]
        outputs = torch.sigmoid(outputs).detach().cpu()
        return outputs[1]

    def batch_gen(self, corpus_pair: list[str]):
        sigmoid = nn.Sigmoid()
        token_input = self.tokenizer(
            corpus_pair,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512)
        dataset = TensorDataset(
            token_input["input_ids"],
            token_input["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch_input, attention_mask = [
                    item.to(self.device) for item in batch]
                outputs = self.model(
                    input_ids=batch_input,
                    attention_mask=attention_mask)
                outputs = sigmoid(outputs)[:, 1]
                preds.append(outputs.detach().cpu())
        preds = torch.cat(preds, dim=0)
        return preds.numpy()


def split2window_str(lines):
    return [''.join(lines[i*10:(i+1)*10]) for i in range(len(lines)//10+1)]

def cal_dep_score(hunk: dict, file_content: str, dependency_analyzer: DependencyClassifier):
    fileB_lines = file_content.splitlines()
    # split file lines into code windows (10 lines)
    hunk_window_str = ''.join(hunk['code_window'])
    code_window_strsB = split2window_str(fileB_lines)
    if len(code_window_strsB) == 0:
        raise KeyError('failed to split fileB into windows')
    # form code windows pairs
    code_window_pairs = [(hunk_window_str, windowB) for windowB in code_window_strsB]
    corpus_pair = dependency_analyzer.construct_corpus_pair(code_window_pairs)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(dependency_analyzer.gen, corpus_pair))
    # get dep score
    dep_score_max = np.max(results).item()
    dep_score_mean = np.mean(results).item()
    dep_score_min = np.min(results).item()
    dep_score_median = np.median(results).item()
    dep_score_std = np.std(results).item()
    return [dep_score_max, dep_score_mean,
            dep_score_min, dep_score_median, dep_score_std]
