"""Per-Pcs baseline."""

import argparse
from collections import defaultdict
import copy
import datetime
import json
import os
from pathlib import Path
import sys
sys.path.append("..")
import time

from accelerate import Accelerator
import numpy as np
from rank_bm25 import BM25Okapi
import torch
from tqdm import tqdm

from data_utils import process_profile
from engine_finetuning import get_tokenizer, load_model, load_generator_from_trained
from eval import evaluate_task_predictions
from merge_utils import find_similar_anchors
from utils import (
    extract_citation_title,
    extract_option,
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
    get_first_k_tokens,
    name2taskid,
    split_batch,
)

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


def get_args_parser():
    parser = argparse.ArgumentParser("LLM p13n per-pcs merging.", add_help=False)
    parser.add_argument("--config_file", default=None, type=str, help="Config yaml file")
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")

    # Model parameters.
    parser.add_argument("--llama_model_path", default=None, type=str, help="path of llama model")
    parser.add_argument("--tokenizer_path", default=None, type=str, help="path of llama model tokenizer")
    parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=3500, metavar="LENGTH", help="the maximum sequence length")
    parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

    # Dataset parameters
    parser.add_argument("--output_dir", default="./output/movie_tagging/LoRA-Composition", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_ckpt", default='./output/movie_tagging/task-base_LLM/lora_ckpt.pt', help="resume lora from checkpoint")
    parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to use gradient checkpoint, recommend TRUE!!")

    parser.add_argument("--anchor_dir", default='./output/movie_tagging/Anchor_PEFT/LoRA', help="resume lora from checkpoint")
    parser.add_argument("--gate_dir", default='./output/movie_tagging/Anchor_PEFT/gate', help="resume lora from checkpoint")
    parser.add_argument("--test_idx_dir", default='./anchor_selection/history_avg/anchor_user_idx.pt', help="resume lora from checkpoint")

    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Generation hyperparameters.
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature")
    
    # This is really only relevant for citation, as the training task differs from the test task.
    parser.add_argument("--train_gen_len", type=int, default=100, help="Max generation length.")
    parser.add_argument("--max_gen_len", type=int, default=100, help="Max generation length.")
    parser.add_argument("--k_list", type=str, default="1,2,4", help="RAG k.")
    parser.add_argument('--infer', default=False, action=argparse.BooleanOptionalAction)

    # Lora composition hyperparameters.
    parser.add_argument("--topk", type=int, default=1, help="top_k")
    parser.add_argument("--recent_k", type=int, default=50, help="No. of recent profile items to use for training.")
    parser.add_argument("--agg_temperature", type=float, default=1, help="temperature")
    parser.add_argument('--sample', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sample_topk", type=int, default=10, help="topk")
    parser.add_argument("--sample_temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--sample_top_p", type=float, default=None, help="Sampling topd p")
    parser.add_argument("--shared_ratio", type=float, default=1, help="shared ratio")

    parser.add_argument("--train_sample", type=float, default=None, help="percentag of training data to sample")
    parser.add_argument("--n_sim", default=None, type=int, help="no. of top similar anchors to use.")
    parser.add_argument("--num_eval_anchors", default=5, type=int, help="no. of top similar anchors to use for eval.")
    parser.add_argument("--user_idxs", default="0-100", type=str, help="for debugging")

    return parser


def process_profile_test_data(user, batch_size, k_list):
    out_list = []
    test_question_list = []
    question_id_list = []
    retrieval_test_question_list = [[] for _ in range(len(k_list))]

    if args.task_name == "citation":
        extract_article = extract_citation_title
    elif args.task_name == "movie_tagging":
        extract_article = extract_movie
    elif args.task_name == "news_categorize":
        extract_article = extract_news_cat
    elif args.task_name == "news_headline":
        extract_article = extract_news_headline
    elif args.task_name == "product_rating":
        extract_article = extract_product_review
    elif args.task_name == "scholarly_title":
        extract_article = extract_scholarly_title
    elif args.task_name == "tweet_paraphrase":
        extract_article = extract_tweet_paraphrasing

    with open("./prompt/prompt.json", "r") as f:
        prompt_template = json.load(f)
        
    user_profile = all_profile[str(user["user_id"])]

    for q in user["query"]:
        if args.task_name == "citation":
            test_question = q["input"]
            test_article = extract_article(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
        else:
            test_question = q["input"]
            test_article = extract_article(test_question)
            test_prompt =  prompt_template[args.task_name]["prompt"].format(test_article)

        test_prompt = f"### User Profile:\n{user_profile}\n\n" + test_prompt

        test_question_list.append(test_prompt)
        question_id_list.append(q["id"])

    visible_history_list = user["profile"]
    for p in visible_history_list:
        for key, value in p.items():
            if isinstance(value, str):
                p[key] = get_first_k_tokens(value, 368)

    history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in visible_history_list]

    tokenized_corpus = [doc.split(" ") for doc in history_list]
    bm25 = BM25Okapi(tokenized_corpus)

    for idx, k in enumerate(k_list):
        for q in user["query"]:
            if args.task_name == "citation":
                test_question = q["input"]
                test_article = extract_article(test_question)
                option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
                test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
            else:
                test_question = q["input"]
                test_article = extract_article(test_question)
                test_prompt =  prompt_template[args.task_name]["prompt"].format(test_article)

            tokenized_query = prompt_template[args.task_name]["retrieval_query_wokey"].format(test_article).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)
        
            history_string = "".join(retrieved_history)

            test_prompt = f"### User History:\n{history_string}\n\n" + test_prompt
            test_prompt = f"### User Profile:\n{user_profile}\n\n" + test_prompt

            retrieval_test_question_list[idx].append(test_prompt)

    test_batch_list = split_batch(test_question_list, batch_size)
    out_list.append(test_batch_list)

    for i, k in enumerate(k_list):
        out_list.append(split_batch(retrieval_test_question_list[i], batch_size))

    all_test_question_list = [test_question_list] + retrieval_test_question_list
    return out_list, question_id_list, all_test_question_list


def process_test_data(user, batch_size, k_list):
    out_list = []
    test_question_list = [] 
    question_id_list = []
    retrieval_test_question_list = [[] for _ in range(len(k_list))]

    if args.task_name == "citation":
        extract_article = extract_citation_title
    elif args.task_name == "movie_tagging":
        extract_article = extract_movie
    elif args.task_name == "news_categorize":
        extract_article = extract_news_cat
    elif args.task_name == "news_headline":
        extract_article = extract_news_headline
    elif args.task_name == "product_rating":
        extract_article = extract_product_review
    elif args.task_name == "scholarly_title":
        extract_article = extract_scholarly_title
    elif args.task_name == "tweet_paraphrase":
        extract_article = extract_tweet_paraphrasing

    with open("./prompt/prompt.json", "r") as f:
        prompt_template = json.load(f)

    for q in user["query"]:
        if args.task_name == "citation":
            test_question = q["input"]
            test_article = extract_article(test_question)
            option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
        else:
            test_question = q["input"]
            test_article = extract_article(test_question)
            test_prompt =  prompt_template[args.task_name]["prompt"].format(test_article)

        test_question_list.append(test_prompt)
        question_id_list.append(q["id"])

    visible_history_list = user["profile"]
    for p in visible_history_list:
        for key, value in p.items():
            if isinstance(value, str):
                p[key] = get_first_k_tokens(value, 368)

    history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in visible_history_list]

    tokenized_corpus = [doc.split(" ") for doc in history_list]
    bm25 = BM25Okapi(tokenized_corpus)

    for idx, k in enumerate(k_list):
        for q in user["query"]:
            if args.task_name == "citation":
                test_question = q["input"]
                test_article = extract_article(test_question)
                option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
                test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
            else:
                test_question = q["input"]
                test_article = extract_article(test_question)
                test_prompt =  prompt_template[args.task_name]["prompt"].format(test_article)

            tokenized_query = prompt_template[args.task_name]["retrieval_query_wokey"].format(test_article).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)
        
            history_string = "".join(retrieved_history)

            test_prompt = f"### User History:\n{history_string}\n\n" + test_prompt

            retrieval_test_question_list[idx].append(test_prompt)

    test_batch_list = split_batch(test_question_list, batch_size)
    out_list.append(test_batch_list)

    for i, k in enumerate(k_list):
        out_list.append(split_batch(retrieval_test_question_list[i], batch_size))

    all_test_question_list = [test_question_list] + retrieval_test_question_list

    return out_list, question_id_list, all_test_question_list


def get_all_history_id(data, tokenizer_path, max_length):
    tokenizer = get_tokenizer(tokenizer_path)
    
    example_all = []
    label_all = []

    for ann in data:
        prompt = ann['prompt']
        example = ann['full_prompt']

        prompt = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        example_all.append(example)

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        label_all.append(labels)
    
    trimmed_examples = [example[:max_length] for example in example_all]
    trimmed_labels = [label[:max_length] for label in label_all]
    
    # Determine the maximum sequence length after trimming but capped at max_length
    max_length = min(max([len(example) for example in trimmed_examples]), max_length)

    # Pad sequences to the determined max_length
    padded_examples = torch.stack([torch.cat((example, torch.zeros(max_length - len(example), dtype=torch.int64) - 1)) if len(example) < max_length else example for example in trimmed_examples])
    padded_labels = torch.stack([torch.cat((label, torch.zeros(max_length - len(label), dtype=torch.int64) - 1)) if len(label) < max_length else label for label in trimmed_labels])

    example_masks = padded_examples.ge(0)
    label_masks = padded_labels.ge(0)

    padded_examples[~example_masks] = 0
    padded_labels[~label_masks] = 0

    return padded_examples, padded_labels


# TODO(kykim): Refactor.
def predict(generator, data_list, task_name,
            batch_size, max_gen_len, temperature, top_p):
    """Computes model predictions on a given data.

    The input data is expected to be a list of dicts, with each dict containing
    the prompt, the full prompt (including the label), and the example id.
    """
    prompts = [datum["prompt"] for datum in data_list]
    train_batch_list = split_batch(prompts, batch_size)

    all_preds = []
    for batch in train_batch_list:
        results = generator.generate(batch, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        all_preds += results

    # A hack to use rouge for train eval in case of citation.
    task_id = "LaMP" if task_name in ("citation") else name2taskid[task_name]
    preds_dict = {"task": task_id, "golds": []}
    for datum, pred in zip(data_list, all_preds):
        output = pred.replace(datum["prompt"], "").strip()
        preds_dict["golds"].append({"id": datum["id"], "output": output})

    return preds_dict


def base_pred(args, model, loras, data_list):
    preds_dicts = []
    for lora in [None] + loras:
        if lora is not None:
            model.get_weighted_lora(lora_corpus=[lora], weights=[1.0])
        generator = load_generator_from_trained(model, args.tokenizer_path)
        max_gen_len = args.train_gen_len if args.task_name == "citation" else args.max_gen_len
        preds_dict = predict(generator, data_list, args.task_name,
                             args.batch_size, max_gen_len, args.temperature, args.top_p)
        preds_dicts.append(preds_dict)
        model.reset_lora_parameters()

    base_preds_dict, lora_preds_dicts = preds_dicts[0], preds_dicts[1:]
    return base_preds_dict, lora_preds_dicts


def merge_eval(preds_dict, lora_preds_dicts, golds_dict):
    eval_result = evaluate_task_predictions(golds_dict, preds_dict)
    lora_eval_results = [
        evaluate_task_predictions(lora_preds_dict, preds_dict)
        for lora_preds_dict in lora_preds_dicts
    ]
    eval_result["debug"] = {
        "lora_eval_results": lora_eval_results,
    }
    return eval_result


def main(args):
    # Fix the seed for reproducibility.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Scratch directory for processes.
    tmp_dir = os.path.join(args.output_dir, "tmp")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write("{}".format(args).replace(", ", ",\n") + "\n")

    # Find top-k similar anchor users.
    anchor_user_ids, topk_sims, topk_idxs = find_similar_anchors(args.task_name, topk=None)
    del topk_sims  # Inspect for debugging purposes.

    accelerator = Accelerator()  # For batch inference.

    # Load the model.
    model = load_model(
        ckpt_dir=args.llama_model_path,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        lora_path=args.lora_ckpt,
        w_lora=args.w_lora,
        grad_ckpt=args.grad_ckpt,
    )
    model.merge_lora_parameters()
    model.set_all_frozen()
    model = accelerator.prepare(model)

    # Load anchor user LoRAs *in the right order*.
    all_loras = []
    for anchor_user_id in anchor_user_ids:
        anchor_dir = os.path.join(args.anchor_dir, f"user_{anchor_user_id}")
        assert os.path.exists(anchor_dir), f"Anchor dir not found: {anchor_dir}"
        lora = torch.load(os.path.join(anchor_dir, "lora_ckpt.pt"), map_location="cpu")
        all_loras.append(lora)

    with open(args.test_data_path, "r") as f:
        test_users = json.load(f)
    user_idxs = args.user_idxs.split("-")
    s_idx, e_idx = int(user_idxs[0]), int(user_idxs[1])
    test_users = test_users[s_idx:e_idx]

    train_golds = dict()
    train_eval_results = dict()
    pred_all = defaultdict(dict)
    pred_profile_all = defaultdict(dict)
    chosen_anchors = defaultdict(list)

    # Test user indices should be computed deterministically prior to split.
    pidx = accelerator.process_index
    test_user_with_idx = [(idx, user) for idx, user in enumerate(test_users)]
    with accelerator.split_between_processes(test_user_with_idx) as sub_test_users:
        for idx, user in tqdm(
            sub_test_users,
            desc=f"({pidx}) Merging",
            total=len(sub_test_users),
        ):
            model.reset_lora_parameters()  # IIUC, this isn't really needed as we override params.

            user_id = str(user["user_id"])
            topk = args.n_sim or len(topk_idxs[idx])

            loras, chosen_anchor_ids = [], []
            for chosen_idx in topk_idxs[idx][:topk]:
                anchor_user_id = str(anchor_user_ids[chosen_idx])
                chosen_anchor_ids.append(anchor_user_id)
                chosen_anchors[f"{user_id}-{idx}"].append(anchor_user_id)
                loras.append(all_loras[chosen_idx])
            lora_path_list = [os.path.join(args.anchor_dir, f"user_{uid}", "lora_ckpt.pt") for uid in chosen_anchor_ids]
            gate_path_list = [os.path.join(args.gate_dir, f"user_{uid}", "gate_ckpt.pt") for uid in chosen_anchor_ids]

            # Evaluate the base model prior to merging.
            data_list = process_profile(user, prompt_template, all_profile, args.task_name, recent_k=args.recent_k)
            if args.train_sample:
                num_samples = int(args.train_sample)
                if args.train_sample <= 1.0:
                    num_samples = int(len(data_list) * args.train_sample)
                data_list = np.random.choice(data_list, size=num_samples, replace=False)

            # A hack to use rouge for train eval in case of citation.
            task_id = "LaMP" if args.task_name in ("citation") else name2taskid[args.task_name]
            golds_dict = {"task": task_id, "golds": []}
            for datum in data_list:
                prompt, full_prompt = datum["prompt"], datum["full_prompt"]
                golds_dict["golds"].append(
                    {
                        "id": datum["id"],
                        "output": full_prompt.replace(prompt, "").strip(),
                    }
                )
            train_golds[user_id] = golds_dict

            train_eval_loras = loras[:args.num_eval_anchors] if args.num_eval_anchors else loras
            base_preds_dict, lora_preds_dicts = base_pred(args, model, train_eval_loras, data_list)
            base_eval_result = merge_eval(base_preds_dict, lora_preds_dicts, golds_dict)

            # Merge and run a final evaluation.
            input_ids, labels = get_all_history_id(data_list, args.tokenizer_path, args.max_seq_len)
            start_time = time.time()
            model.get_new_lora(
                lora_path_list=lora_path_list,
                gate_path_list=gate_path_list,
                input_ids=input_ids,
                labels=labels,
                batch_size=args.batch_size,
                topk=args.topk,
                epoch=args.epochs,
                temperature=args.agg_temperature,
                sample=args.sample,
                sample_topk=args.sample_topk,
                sample_temperature=args.sample_temperature,
                sample_top_p=args.sample_top_p,
                shared_ratio=args.shared_ratio
            )
            total_time = time.time() - start_time
            user_out_dir = os.path.join(args.output_dir, "loras", f"user_{user_id}")
            Path(user_out_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.lora_state_dict(), os.path.join(user_out_dir, "lora_ckpt.pt"))
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Selecting time {}".format(total_time_str))

            generator = load_generator_from_trained(model, args.tokenizer_path)
            max_gen_len = args.train_gen_len if args.task_name == "citation" else args.max_gen_len
            preds_dict = predict(generator, data_list, args.task_name,
                                 args.batch_size, max_gen_len, args.temperature, args.top_p)
            eval_result = merge_eval(preds_dict, lora_preds_dicts, golds_dict)
            train_eval_results[user_id] = {"base": base_eval_result, "final": eval_result}
            print(f"Eval results: {train_eval_results[user_id]}")

            # Inference stage.
            # TODO(kykim): Clean up.
            generator = load_generator_from_trained(model, args.tokenizer_path)
            test_batch_list, test_id_list, test_question_list = process_test_data(user, batch_size=args.batch_size, k_list=args.k_list)
            for idx, setting in enumerate(test_batch_list):
                all_results = []
                for batch in setting:
                    results = generator.generate(batch, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
                    all_results += results
                pred_user = []
                for i in range(len(all_results)):
                    output = all_results[i].replace(test_question_list[idx][i], "").strip()
                    pred_user.append({"id": test_id_list[i], "output": output})
                k_idx = 0 if idx == 0 else args.k_list[idx-1]
                pred_all[k_idx][user_id] = pred_user

            test_batch_list, test_id_list, test_question_list = process_profile_test_data(user, batch_size=args.batch_size, k_list=args.k_list)
            for idx, setting in enumerate(test_batch_list):
                all_results = []
                for batch in setting:
                    results = generator.generate(batch, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
                    all_results += results
                pred_profile_user = []
                for i in range(len(all_results)):
                    output = all_results[i].replace(test_question_list[idx][i], "").strip()
                    pred_profile_user.append({"id": test_id_list[i], "output": output})
                k_idx = 0 if idx == 0 else args.k_list[idx-1]
                pred_profile_all[k_idx][user_id] = pred_profile_user

        # Write out per-process results.
        for fname, data_dict in zip(
            ["train-golds.json", "train-evals.json", "pred.json", "pred-profile.json", "chosen-anchors.json"],
            [train_golds, train_eval_results, pred_all, pred_profile_all, chosen_anchors],
        ):
            with open(os.path.join(tmp_dir, f"{pidx}-{fname}"), "w") as f:
                json.dump(data_dict, f, indent=4)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.test_data_path = f"./data/{args.task_name}/test_100/user_test_100.json"
    args.test_gold_data_path = f"./data/{args.task_name}/test_100/user_test_100_label.json"
    args.k_list = [int(k) for k in args.k_list.split(",")]
    print(args)
    # TODO(kykim): Fix this hack.
    with open(f'./data/{args.task_name}/profile-id2text.json', 'r') as f:
        all_profile = json.load(f)
    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)
    main(args)
