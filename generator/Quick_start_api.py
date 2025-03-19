import torch
import argparse
import torch.nn as nn

from run import InputFeatures
from model import Seq2Seq
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


def construct_input(
    code_window: list[str], edit_labels: list[str], prompt: str, prior_edits: list[dict]
) -> str:
    """
    Constructs the input string for the line locator model.
    """
    input_str = ""
    for label, line in zip(edit_labels, code_window):
        input_str += f'{label} {line}'
    input_str += f'</s>{prompt}</s>'
    for edit in prior_edits:
        if edit['before'] != "":
            input_str += f"remove {edit['before']} </s> add{edit['after']}</s>"
    return input_str


def convert_examples_to_features(input_seq, tokenizer, args):
    features = []
    # source
    source_tokens = tokenizer.tokenize(input_seq)[: args.max_source_length - 2]

    # the reset is the same as original code
    source_tokens = [tokenizer.cls_token] + \
        source_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens))
    padding_length = args.max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length

    # target
    target_tokens = tokenizer.tokenize('None')
    target_tokens = [tokenizer.cls_token] + \
        target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] * len(target_ids)
    padding_length = args.max_target_length - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length
    target_mask += [0] * padding_length

    features.append(InputFeatures(
        0, source_ids, target_ids, source_mask, target_mask,))

    return features


def generator_api(
    code_window: list[str],
    edit_labels: list[str],
    prompt: str,
    prior_edits: list[dict],
    language: str,
) -> list[str]:
    assert language in ['python', 'go', 'java', 'javascript', 'typescript']
    config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    encoder = RobertaModel.from_pretrained(
        'microsoft/codebert-base', config=config)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.max_source_length = 512
    args.max_target_length = 512
    args.beam_size = 10

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads
    )
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        config=config,
        beam_size=args.beam_size,
        max_length=args.max_target_length,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
    )
    model.load_state_dict(
        torch.load(
            f'model/{language}/checkpoint-best-bleu/pytorch_model.bin', map_location=device,
        )
    )

    input_str = construct_input(code_window, edit_labels, prompt, prior_edits)
    inter_features = convert_examples_to_features(input_str, tokenizer, args)
    all_source_ids = torch.tensor(
        [f.source_ids for f in inter_features], dtype=torch.long)
    all_source_mask = torch.tensor(
        [f.source_mask for f in inter_features], dtype=torch.long)
    with torch.no_grad():
        preds = model(source_ids=all_source_ids,
                      source_mask=all_source_mask, args=args)
        pred = preds[0]
        multiple_results = []
        for candidate in pred:
            t = candidate.cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[: t.index(0)]
            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
            multiple_results.append(text)

    return multiple_results
