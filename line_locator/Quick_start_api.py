import os
import torch
import argparse
import requests

from model import Seq2Seq
from run import Example, convert_examples_to_features
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

def construct_input(
    code_window: list[str], prompt: str, prior_edits: list[dict]
) -> str:
    """
    Constructs the input string for the line locator model.
    """
    input_str = ""
    for line in code_window:
        input_str += f"<mask> {line}"
    input_str += f"</s> {prompt} </s>"
    for edit in prior_edits:
        if edit["code_before"] != []:
            input_str += f"remove {''.join(edit['code_before'])} </s> add {''.join(edit['code_after'])}</s>"
        else:
            input_str += f"add {''.join(edit['code_after'])}</s>"
    return input_str


def line_locator_api(
    code_window: list[str], prompt: str, prior_edits: list[dict], language: str
) -> list[str]:
    assert language in ["python", "go", "java", "javascript", "typescript"]
    config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    encoder = RobertaModel.from_pretrained("microsoft/codebert-base", config=config)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.max_source_length = 512
    args.max_target_length = 512

    model = Seq2Seq(
        encoder=encoder,
        config=config,
        beam_size=10,
        max_length=512,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
        mask_id=tokenizer.mask_token_id,
    )
    if not os.path.exists(f"model/{language}/locator_model.bin"):
        # download model from huggingface
        os.makedirs(f"model/{language}", exist_ok=True)
        download("model", language)

    model.load_state_dict(
        torch.load(
            f"model/{language}/locator_model.bin",
            map_location=device,
        )
    )

    input_str = construct_input(code_window, prompt, prior_edits)
    examples = [Example(idx=0, source=input_str, target=[""] * len(code_window))]

    inter_features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        stage="test",
        args=args,
    )
    all_source_ids = torch.tensor(
        [f.source_ids for f in inter_features], dtype=torch.long
    )
    all_source_mask = torch.tensor(
        [f.source_mask for f in inter_features], dtype=torch.long
    )
    all_target_ids = torch.tensor(
        [f.target_ids for f in inter_features], dtype=torch.long
    )
    all_target_mask = torch.tensor(
        [f.target_mask for f in inter_features], dtype=torch.long
    )

    lm_logits = model(
        source_ids=all_source_ids,
        source_mask=all_source_mask,
        target_ids=all_target_ids,
        target_mask=all_target_mask,
        train=False,
    ).to("cpu")

    output = []
    for j in range(lm_logits.shape[1]):  # for every token
        if all_source_ids[0][j] == tokenizer.mask_token_id:  # if is masked
            output.append(
                tokenizer.decode(
                    torch.argmax(lm_logits[0][j]),
                    clean_up_tokenization_spaces=False,
                )
            )

    return output

def download(model_dir: str, lang: str) -> int:
    lang_model_dir = f'{model_dir}/{lang}'

    generator_model_base_url = f'https://huggingface.co/code-philia/CoEdPilot-line-locator/resolve/main/{lang}'

    download_list = [
        [f'{generator_model_base_url}/checkpoint-best-bleu/pytorch_model.bin',
            'locator_model.bin'],
    ]
    for it in download_list:
        print(f'Cloning models for \'{lang}\' to {lang_model_dir}/{it[1]}...')
        res = download_file(
            url=it[0],
            target=f'{lang_model_dir}/{it[1]}'
        )
        if res != 0:
            return 2
        print(f'{it[1]} downloaded.')

    print(f'All models for {lang} is ready.')

    return 0

def download_file(url: str, target: str) -> int:
    """
    Download a single file from `url`, save to file `target`.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download: {url}")
        return 1
    with open(target, 'wb') as f:
        f.write(response.content)

    return 0

if __name__ == "__main__":
    code_window = ["def hello_world():", "    print('Hello, World!')"]
    prompt = "add a new line"
    prior_edits = [{"code_before": [], "code_after": ["print('Hello, World!')"]}]
    language = "python"
    line_locator_api(code_window, prompt, prior_edits, language)
