# dataset_generator

指定输入 token 长度和数据条数，生成数据集用于 vLLM benchmark 性能测试。

输出格式为 vLLM `custom` 数据集的 JSONL 格式（`{"prompt": "...", "output_tokens": N}`），可直接用 `vllm bench serve --dataset-name custom` 加载，无长度限制。

## 目录结构

```
dataset_generator/
├── generate.py          # 主生成脚本
├── data_picker.py       # 数据选择器
├── config.py            # 配置文件
├── sources/             # 数据源目录（需自行下载，见下方说明）
└── output/              # 生成数据集的输出目录
```

## 一、准备数据源

数据源文件未包含在仓库中，需自行下载到 `sources/` 目录下：

### GSM8K 数据源

```bash
mkdir -p sources
# 从 HuggingFace 下载 GSM8K 数据集的 test 部分并转换为 jsonl 格式
# 每行格式: {"question": "...", "answer": "..."}
# 下载后放置为 sources/GSM8K.jsonl（1319条）
```

GSM8K 数据可以从 [HuggingFace](https://huggingface.co/datasets/gsm8k) 获取，需要将 `question` 和 `answer` 字段提取为 JSONL 格式。

### ShareGPT 数据源

```bash
mkdir -p sources/sharegpt
# 从 ModelScope 下载
git clone https://www.modelscope.cn/datasets/huangjintao/sharegpt.git
# 将所需文件复制到 sources/sharegpt/ 目录
cp sharegpt/computer_en_26k.jsonl sources/sharegpt/
```

建议使用 `computer_en_26k.jsonl`（20692条英文计算机类对话），如需中文数据可选用 `computer_zh_26k.jsonl` 或 `unknow_zh_38k.jsonl`。

下载完成后，修改 `config.py` 中对应路径即可。

## 二、修改 config.py

使用前修改 `config.py`，填入模型权重路径（HuggingFace 格式，需包含 `tokenizer.json`、`tokenizer_config.json`）：

```python
TOKENIZER_PATH = "/home/weights/model_weights"   # 改成你的模型路径
```

## 二、生成数据集

```bash
# 生成 ShareGPT 数据，输入 2048 tokens，输出 2048 tokens，100 条
python generate.py --dataset_type sharegpt --input_len 2048 --output_len 2048 --data_num 100

# 生成 GSM8K 数据，输入 1024 tokens，输出 512 tokens，500 条
python generate.py --dataset_type gsm8k --input_len 1024 --output_len 512 --data_num 500

# 不指定 output_len 时，vLLM benchmark 用 --custom-output-len 控制输出长度
python generate.py --dataset_type sharegpt --input_len 2048 --data_num 100

# 临时覆盖 tokenizer 路径
python generate.py --tokenizer_path /other/model/path --dataset_type sharegpt --input_len 2048 --data_num 100
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset_type` | 数据集类型：`gsm8k` 或 `sharegpt` | 必填 |
| `--input_len` | 输入 prompt 的 token 长度 | 必填 |
| `--output_len` | 输出 token 长度（写入数据集 `output_tokens` 字段） | None |
| `--data_num` | 数据条数 | 必填 |
| `--tokenizer_path` | 模型权重路径 | 读 config.py |
| `--output_dir` | 输出目录 | 读 config.py |
| `--gsm8k_path` | GSM8K 数据源路径 | 读 config.py |
| `--sharegpt_path` | ShareGPT 数据源路径 | 读 config.py |
| `--no_repeat` | 不重复选取数据源 | False |
| `--seed` | 随机种子 | None |

### 输出格式

生成的数据集为 JSONL 格式，每行一个 JSON，兼容 vLLM `custom` 数据集：

```jsonl
{"prompt": "调整到 input_len 长度的文本", "output_tokens": 2048}
{"prompt": "另一条文本", "output_tokens": 2048}
```

- `prompt`：调整到指定 `input_len` token 长度的文本
- `output_tokens`：指定输出 token 长度（仅当传入 `--output_len` 时写入）

输出文件自动命名如 `ShareGPT-in2048-out2048-num100.jsonl`。

## 三、用于 vLLM benchmark

### 为什么用 `custom` 而不是 `sharegpt` 数据集类型？

vLLM 的 `sharegpt` 数据集类型有两个限制：
1. 读 JSON 数组文件（`json.load`），不支持 JSONL
2. **硬编码长度过滤**：`max_prompt_len=1024`、`max_total_len=2048`，超过 1024 tokens 的 prompt 会被丢弃，无法测 2k/4k/8k 场景

`custom` 数据集类型没有这些限制，读 JSONL，不限 prompt 长度，且支持每行指定 `output_tokens`。

### 输入和输出长度的控制方式

| 长度 | 控制方式 | 说明 |
| --- | --- | --- |
| 输入长度 | 数据集 `--input_len` | prompt 的 token 数（不含 chat template） |
| 输出长度 | `--custom-output-len` 或 `--output-len` | vLLM benchmark 命令中指定 |

> **注意**：vLLM `custom` 数据集默认会对 prompt 应用 `tokenizer.apply_chat_template`，这会额外添加 chat template tokens（如 `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`）。如果希望**精确控制输入 token 数**（含 chat template），加上 `--skip-chat-template` 跳过模板；否则实际输入长度 = `input_len` + chat template overhead。

### vLLM bench serve 使用示例

先启动 vLLM 服务：

```bash
vllm serve /home/weights/model_weights --port 8000
```

跑 benchmark：

```bash
# 场景1：ShareGPT 2k输入 / 2k输出（输出长度写在数据集 output_tokens 字段中）
python generate.py --dataset_type sharegpt --input_len 2048 --output_len 2048 --data_num 100
vllm bench serve \
    --base-url http://localhost:8000 \
    --model model_name \
    --dataset-name custom \
    --dataset-path output/ShareGPT-in2048-out2048-num100.jsonl \
    --custom-output-len -1 \
    --num-prompts 100

# 场景2：输出长度不在数据集中写，由 benchmark 命令统一指定
python generate.py --dataset_type sharegpt --input_len 2048 --data_num 100
vllm bench serve \
    --base-url http://localhost:8000 \
    --model model_name \
    --dataset-name custom \
    --dataset-path output/ShareGPT-in2048-num100.jsonl \
    --custom-output-len 2048 \
    --num-prompts 100

# 场景3：精确控制输入长度（跳过 chat template，避免额外 token）
vllm bench serve \
    --base-url http://localhost:8000 \
    --model model_name \
    --dataset-name custom \
    --dataset-path output/ShareGPT-in2048-out2048-num100.jsonl \
    --custom-output-len -1 \
    --skip-chat-template \
    --num-prompts 100

# 场景4：GSM8K 数据
python generate.py --dataset_type gsm8k --input_len 1024 --output_len 512 --data_num 50
vllm bench serve \
    --base-url http://localhost:8000 \
    --model model_name \
    --dataset-name custom \
    --dataset-path output/GSM8K-in1024-out512-num50.jsonl \
    --custom-output-len -1 \
    --num-prompts 50
```

### `--custom-output-len` 参数说明

| 值 | 含义 |
| --- | --- |
| 正整数（如 `2048`） | 所有请求统一使用该输出长度，忽略数据集中的 `output_tokens` |
| `-1` | 每条请求使用数据集自身的 `output_tokens` 字段值（需数据集包含该字段） |

## 四、vLLM 自带的指定长度测试方式

vLLM 内置了 `random` 数据集类型，可以直接指定输入和输出长度，不需要生成数据集文件：

```bash
# 直接用 random 数据集测 2k输入/2k输出
vllm bench serve \
    --base-url http://localhost:8000 \
    --model model_name \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 2048 \
    --num-prompts 100
```

**区别**：
- `random`：生成随机 token 序列作为 prompt，内容无意义，但可以精确控制长度
- `custom`（本工具生成的）：使用真实文本作为 prompt，更贴近实际推理场景

| 方式 | 输入内容 | 输入长度 | 输出长度 | 适用场景 |
| --- | --- | --- | --- | --- |
| `random` | 随机 tokens | `--random-input-len` | `--random-output-len` | 纯性能测试，只关心吞吐 |
| `custom`（本工具） | 真实文本 | `--input_len` | `--output_len` | 性能测试 + 更真实的推理场景 |

## 五、数据生成逻辑

从数据源随机抽取文本，用 tokenizer tokenize 后截断或重复填充到 `input_len`：
- 原文 tokens >= input_len：截断
- 原文 tokens < input_len：重复填充后再截断

- ShareGPT：取对话中所有 human 发言拼接为 prompt
- GSM8K：取 question 字段作为 prompt

## 六、常见问题

**1. 加载 tokenizer 报错**

检查 transformers 版本是否适配模型，如 GLM5 需更新 transformers。

**2. 数据生成数量不足**

GSM8K 数据源只有 1319 条，开 `--no_repeat` 且 `--data_num > 1319` 会失败。关闭即可重复使用。ShareGPT 有 20692 条，一般不会不够。

**3. 想用中文 ShareGPT 数据**

修改 `config.py` 中的 `SHAREGPT_PATH`，指向中文数据源文件即可。

**4. 实际输入长度和指定的 input_len 不一致**

可能是因为 vLLM 默认会 apply chat template。加上 `--skip-chat-template` 可跳过模板，让实际输入长度等于 `input_len`。

**5. vLLM sharegpt 数据集类型过滤了我的数据**

vLLM 的 `sharegpt` 类型有 `max_prompt_len=1024` 硬编码限制。本工具输出 `custom` 格式，用 `--dataset-name custom` 加载即可，无此限制。
