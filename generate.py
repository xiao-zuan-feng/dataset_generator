import argparse
import json
import logging
import os
import random

from transformers import AutoTokenizer
from data_picker import GSM8KPicker, ShareGPTPicker
from config import TOKENIZER_PATH, OUTPUT_DIR, GSM8K_PATH, SHAREGPT_PATH

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def truncate_or_pad(tokenizer, text, target_len):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) >= target_len:
        tokens = tokens[:target_len]
    else:
        repeat_times = (target_len + len(tokens) - 1) // len(tokens)
        tokens = (tokens * repeat_times)[:target_len]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def adjust_text_length(tokenizer, text, target_len):
    adjusted = truncate_or_pad(tokenizer, text, target_len)
    verify_len = len(tokenizer.encode(adjusted, add_special_tokens=False))
    if verify_len != target_len:
        corrected_tokens = tokenizer.encode(adjusted, add_special_tokens=False)
        if len(corrected_tokens) >= target_len:
            corrected_tokens = corrected_tokens[:target_len]
        else:
            corrected_tokens = (corrected_tokens * ((target_len // len(corrected_tokens)) + 1))[:target_len]
        adjusted = tokenizer.decode(corrected_tokens, skip_special_tokens=True)
    return adjusted


def generate_gsm8k_dataset(tokenizer, input_len, data_num, gsm8k_path):
    picker = GSM8KPicker(gsm8k_path, no_repeat=False)
    samples = []
    pbar = tqdm(total=data_num, desc="Generating GSM8K dataset", unit="row") if tqdm else None

    for _ in range(data_num):
        raw_text = picker.pick_one()
        prompt = adjust_text_length(tokenizer, raw_text, input_len)
        samples.append({"prompt": prompt})
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()
    return samples


def generate_sharegpt_dataset(tokenizer, input_len, data_num, sharegpt_path):
    picker = ShareGPTPicker(sharegpt_path, no_repeat=False)
    samples = []
    pbar = tqdm(total=data_num, desc="Generating ShareGPT dataset", unit="row") if tqdm else None

    for _ in range(data_num):
        raw_data = picker.pick_one()
        conversations = raw_data.get("conversations", raw_data.get("conversation", []))

        human_parts = []
        for turn in conversations:
            human_val = turn.get("human", turn.get("from_human", turn.get("value", "")))
            if human_val and isinstance(human_val, str):
                human_parts.append(human_val)

        if not human_parts:
            continue

        prompt = adjust_text_length(tokenizer, " ".join(human_parts), input_len)
        samples.append({"prompt": prompt})
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()
    return samples


def write_jsonl(samples, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info(f"数据集已写入: {output_path} (共 {len(samples)} 条)")


def main():
    parser = argparse.ArgumentParser(description="数据集生成工具 - 生成 vLLM benchmark custom 格式数据")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="模型 tokenizer 路径（默认读 config.py）")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["gsm8k", "sharegpt"], help="数据集类型")
    parser.add_argument("--input_len", type=int, required=True, help="输入 prompt 的 token 长度")
    parser.add_argument("--data_num", type=int, required=True, help="生成数据条数")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认读 config.py）")
    parser.add_argument("--gsm8k_path", type=str, default=None, help="GSM8K 数据源路径（默认读 config.py）")
    parser.add_argument("--sharegpt_path", type=str, default=None, help="ShareGPT 数据源路径（默认读 config.py）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or TOKENIZER_PATH
    output_dir = args.output_dir or OUTPUT_DIR
    gsm8k_path = args.gsm8k_path or GSM8K_PATH
    sharegpt_path = args.sharegpt_path or SHAREGPT_PATH

    if args.seed is not None:
        random.seed(args.seed)

    logging.info(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if args.dataset_type == "gsm8k":
        samples = generate_gsm8k_dataset(tokenizer, args.input_len, args.data_num, gsm8k_path)
        ds_tag = "GSM8K"
    else:
        samples = generate_sharegpt_dataset(tokenizer, args.input_len, args.data_num, sharegpt_path)
        ds_tag = "ShareGPT"

    filename = f"{ds_tag}-in{args.input_len}-num{len(samples)}.jsonl"
    output_path = os.path.join(output_dir, filename)
    write_jsonl(samples, output_path)

    logging.info(f"生成完成: {len(samples)} 条数据, 输入长度 {args.input_len} tokens")
    logging.info(f"输出文件: {output_path}")
    logging.info(f"vLLM benchmark 使用方式: vllm bench serve --dataset-name custom --dataset-path {output_path} --custom-output-len <输出长度>")


if __name__ == "__main__":
    main()
