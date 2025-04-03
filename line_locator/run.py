# coding=utf-8
from __future__ import absolute_import

import argparse
import json
import logging
import numpy as np
import os
import random

import torch
from io import open

from model import Seq2Seq
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer
)

import bleu

MODEL_CLASSES = {"roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if "idx" not in js:
                js["idx"] = idx
            code = js["code_tokens"]
            label = js["docstring_tokens"].split(" ")

            if len(label) != code.count("<mask>"):
                continue
            examples.append(Example(idx=idx, source=code, target=label))
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, example_id, source_ids, target_ids, source_mask, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples)):
        # source
        source_tokens = tokenizer.tokenize(example.source)[: args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        original_source_len = len(source_ids)
        source_mask = [1] * len(source_tokens)
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        target_tokens = tokenizer.tokenize(example.source)[: args.max_target_length - 2]
        label_idx = 0
        # replace mask token with label token
        for i in range(len(target_tokens)):
            if target_tokens[i] == tokenizer.mask_token:
                target_tokens[i] = example.target[label_idx]
                label_idx += 1

        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        original_target_len = len(target_ids)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if original_source_len != original_target_len:
            print(example.source)
            print(example.target)
            print("source length: ", original_source_len)
            print("target length: ", original_target_len)
            break

        if example_index < 1:
            if stage == "train":
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info(
                    "source_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in source_tokens]
                    )
                )
                logger.info("source_ids: {}".format(" ".join(map(str, source_ids))))
                logger.info("source_mask: {}".format(" ".join(map(str, source_mask))))

                logger.info(
                    "target_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in target_tokens]
                    )
                )
                logger.info("target_ids: {}".format(" ".join(map(str, target_ids))))
                logger.info("target_mask: {}".format(" ".join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type: e.g. roberta",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model: e.g. roberta-base",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        help="Path to trained model: Should contain the .bin files",
    )

    # Other parameters
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=32,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--beam_size", default=10, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--eval_steps", default=-1, type=int, help="")
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    # Parse arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )
    args.device = device

    # Set seed
    set_seed(args.seed)

    # Make directory if output_dir does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )

    # Build model
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    model = Seq2Seq(
        encoder=encoder,
        config=config,
        beam_size=args.beam_size,
        max_length=args.max_target_length,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
        mask_id=tokenizer.mask_token_id,
    )
    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args, stage="train"
        )
        all_source_ids = torch.tensor(
            [f.source_ids for f in train_features], dtype=torch.long
        )
        all_source_mask = torch.tensor(
            [f.source_mask for f in train_features], dtype=torch.long
        )
        all_target_ids = torch.tensor(
            [f.target_ids for f in train_features], dtype=torch.long
        )
        all_target_mask = torch.tensor(
            [f.target_mask for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_source_ids, all_source_mask, all_target_ids, all_target_mask
        )

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size // args.gradient_accumulation_steps,
        )

        num_train_optimization_steps = args.train_steps

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = (
            0,
            0,
            0,
            0,
            0,
            1e6,
        )
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = model(
                    source_ids=source_ids,
                    source_mask=source_mask,
                    target_ids=target_ids,
                    target_mask=target_mask,
                )

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(
                    tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4
                )
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval:
                eval_flag = False
                if "dev_loss" in dev_dataset:
                    eval_examples, eval_data = dev_dataset["dev_loss"]
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(
                        eval_examples, tokenizer, args, stage="dev"
                    )
                    all_source_ids = torch.tensor(
                        [f.source_ids for f in eval_features], dtype=torch.long
                    )
                    all_source_mask = torch.tensor(
                        [f.source_mask for f in eval_features], dtype=torch.long
                    )
                    all_target_ids = torch.tensor(
                        [f.target_ids for f in eval_features], dtype=torch.long
                    )
                    all_target_mask = torch.tensor(
                        [f.target_mask for f in eval_features], dtype=torch.long
                    )
                    eval_data = TensorDataset(
                        all_source_ids, all_source_mask, all_target_ids, all_target_mask
                    )
                    dev_dataset["dev_loss"] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data,
                    sampler=eval_sampler,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                )

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = model(
                            source_ids=source_ids,
                            source_mask=source_mask,
                            target_ids=target_ids,
                            target_mask=target_mask,
                        )
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {
                    "eval_ppl": round(np.exp(eval_loss), 5),
                    "global_step": global_step + 1,
                    "train_loss": round(train_loss, 5),
                }
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                last_output_dir = os.path.join(args.output_dir, "checkpoint-last")
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss
                    output_dir = os.path.join(args.output_dir, "checkpoint-best-ppl")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                if "dev_bleu" in dev_dataset:
                    eval_examples, eval_data = dev_dataset["dev_bleu"]
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(
                        eval_examples, min(1000, len(eval_examples))
                    )
                    eval_features = convert_examples_to_features(
                        eval_examples, tokenizer, args, stage="test"
                    )
                    all_source_ids = torch.tensor(
                        [f.source_ids for f in eval_features], dtype=torch.long
                    )
                    all_source_mask = torch.tensor(
                        [f.source_mask for f in eval_features], dtype=torch.long
                    )
                    all_target_ids = torch.tensor(
                        [f.target_ids for f in eval_features], dtype=torch.long
                    )
                    all_target_mask = torch.tensor(
                        [f.target_mask for f in eval_features], dtype=torch.long
                    )
                    eval_data = TensorDataset(
                        all_source_ids, all_source_mask, all_target_ids, all_target_mask
                    )
                    dev_dataset["dev_bleu"] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data,
                    sampler=eval_sampler,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                )

                model.eval()

                outputs = []
                gts = []
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch
                    with torch.no_grad():
                        lm_logits = model(
                            source_ids=source_ids,
                            source_mask=source_mask,
                            target_ids=target_ids,
                            target_mask=target_mask,
                            train=False,
                        ).to("cpu")
                        # extract masked edit operations
                        # for sample within batch
                        for i in range(lm_logits.shape[0]):
                            output = []
                            gt = []
                            # for every token
                            for j in range(lm_logits.shape[1]):
                                if (
                                    source_ids[i][j] == tokenizer.mask_token_id
                                ):  # if is masked
                                    output.append(
                                        tokenizer.decode(
                                            torch.argmax(lm_logits[i][j]),
                                            clean_up_tokenization_spaces=False,
                                        )
                                    )
                                    gt.append(
                                        tokenizer.decode(
                                            target_ids[i][j],
                                            clean_up_tokenization_spaces=False,
                                        )
                                    )
                            outputs.append(" ".join(output))
                            gts.append(" ".join(gt))
                model.train()
                predictions = []
                with open(os.path.join(args.output_dir, "dev.output"), "w") as f, open(
                    os.path.join(args.output_dir, "dev.gold"), "w"
                ) as f1:
                    for i, (ref, gold) in enumerate(zip(outputs, gts)):
                        predictions.append(str(i) + "\t" + ref)
                        f.write(str(i) + "\t" + ref + "\n")
                        f1.write(str(i) + "\t" + gold + "\n")

                (goldMap, predictionMap) = bleu.computeMaps(
                    predictions, os.path.join(args.output_dir, "dev.gold")
                )
                dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, "checkpoint-best-bleu")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        files = []
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(
                eval_examples, tokenizer, args, stage="test"
            )
            all_source_ids = torch.tensor(
                [f.source_ids for f in eval_features], dtype=torch.long
            )
            all_source_mask = torch.tensor(
                [f.source_mask for f in eval_features], dtype=torch.long
            )
            all_target_ids = torch.tensor(
                [f.target_ids for f in eval_features], dtype=torch.long
            )
            all_target_mask = torch.tensor(
                [f.target_mask for f in eval_features], dtype=torch.long
            )
            eval_data = TensorDataset(
                all_source_ids, all_source_mask, all_target_ids, all_target_mask
            )

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data,
                sampler=eval_sampler,
                batch_size=args.eval_batch_size,
                shuffle=False,
            )

            model.eval()
            outputs = []
            gts = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                with torch.no_grad():
                    lm_logits = model(
                        source_ids=source_ids,
                        source_mask=source_mask,
                        target_ids=target_ids,
                        target_mask=target_mask,
                        train=False,
                    ).to("cpu")
                    # extract masked edit operations
                    # for sample within batch
                    for i in range(lm_logits.shape[0]):
                        output = []
                        gt = []
                        for j in range(lm_logits.shape[1]):  # for every token
                            if (
                                source_ids[i][j] == tokenizer.mask_token_id
                            ):  # if is masked
                                output.append(
                                    tokenizer.decode(
                                        torch.argmax(lm_logits[i][j]),
                                        clean_up_tokenization_spaces=False,
                                    )
                                )
                                gt.append(
                                    tokenizer.decode(
                                        target_ids[i][j],
                                        clean_up_tokenization_spaces=False,
                                    )
                                )
                        outputs.append(" ".join(output))
                        gts.append(" ".join(gt))
            model.train()
            predictions = []
            with open(
                os.path.join(args.output_dir, "test_{}.output".format(str(idx))), "w"
            ) as f, open(
                os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), "w"
            ) as f1:
                for i, (ref, gold) in enumerate(zip(outputs, gts)):
                    predictions.append(str(i) + "\t" + ref)
                    f.write(str(i) + "\t" + ref + "\n")
                    f1.write(str(i) + "\t" + gold + "\n")

            (goldMap, predictionMap) = bleu.computeMaps(
                predictions,
                os.path.join(args.output_dir, "test_{}.gold".format(str(idx))),
            )
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()
