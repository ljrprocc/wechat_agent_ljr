# MLX / mlx-lm 路线规划

## 目标

为 Apple Silicon 机器提供一条比 `PyTorch/Transformers + MPS` 更稳的本地 GPU 推理路线。

当前建议链路：

`Android App -> backend/api_server.py -> backend/agent.py or future backend/mlx_agent.py -> mlx-lm -> Apple Silicon GPU`

## 背景

当前项目已经验证：

- `CPU` 可以跑通 v1.0
- `PyTorch/Transformers + MPS` 在当前机器上会触发 `MPSTemporaryNDArray ... total bytes of NDArray > 2**32`

因此，在 Apple Silicon 上继续追 GPU，本项目更适合评估 `MLX / mlx-lm`。

## 参考仓库

本路线主要基于 Apple 官方开源仓库和 Qwen 官方文档：

- Apple `MLX`：
  https://github.com/ml-explore/mlx
- Apple `mlx-lm`：
  https://github.com/ml-explore/mlx-lm
- Qwen 官方 `MLX LM` 文档：
  https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html

## 为什么考虑 MLX

根据 Apple 官方 `MLX` 仓库：

- `MLX` 是 Apple 为 Apple silicon 提供的机器学习数组框架
- 支持 CPU 和 GPU
- 采用 unified memory

根据 Apple 官方 `mlx-lm` 仓库：

- `mlx-lm` 是在 Apple silicon 上运行和微调 LLM 的 Python 包
- 提供 `load`、`generate`、`chat`、`convert` 等能力
- 可以直接从 Hugging Face Hub 使用兼容模型

这些特性更贴近当前项目在 MacBook 上的实验目标。

## 对 Qwen 的意义

Qwen 官方文档已经提供了 `mlx-lm` 的使用方式，并说明：

- 可以直接加载 Qwen 的 MLX 格式模型
- 也可以把 Hugging Face 模型转换成 MLX 格式
- 官方示例里通过 `tokenizer_config={"eos_token": "<|im_end|>"}` 显式传入结束符

示例思路是：

```python
from mlx_lm import load, generate

model, tokenizer = load("Qwen/Qwen2.5-7B-Instruct-MLX")
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=text, max_tokens=512)
```

这说明对本项目来说，Qwen 路线在 `mlx-lm` 上是有官方材料可以参考的。

## 推荐接入方式

### 阶段 1：文档与独立实验

目标：

- 在不影响 v1.0 CPU 主线的前提下，单独验证 `mlx-lm` 能否稳定完成 Qwen 推理

建议动作：

1. 先安装单独的 MLX 依赖，不影响现有 `requirements.txt`
2. 使用 `experiments/mlx_qwen_smoke.py` 跑一条最小问答
3. 优先验证 `Qwen2.5-0.5B`，确认 Apple GPU 路线可用后再测 `3B`

当前仓库里的实验脚本会优先使用本地已转换好的 `./mlx/...` 目录；只有本地目录不存在时，才回退到 `mlx-community/...` 仓库。

当前仓库已经补了以下实验入口：

- [experiments/mlx_qwen_smoke.py](/Users/lijingru/code/qwen-2.5/experiments/mlx_qwen_smoke.py)
- [requirements-mlx.txt](/Users/lijingru/code/qwen-2.5/requirements-mlx.txt)

建议命令：

```bash
conda activate app
cd /Users/lijingru/code/qwen-2.5

pip install -r requirements-mlx.txt
python experiments/mlx_qwen_smoke.py --model-id qwen2.5-0.5b --prompt "请用三句话介绍你自己"
```

如果你想直接指定具体的 MLX 仓库或本地转换目录，也可以：

```bash
python experiments/mlx_qwen_smoke.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --eos-token "<|im_end|>" \
  --prompt "请用三句话介绍你自己"
```

或者：

```bash
python experiments/mlx_qwen_smoke.py \
  --model ./mlx/Qwen2.5-0.5B-Instruct-4bit \
  --eos-token "<|im_end|>" \
  --prompt "请用三句话介绍你自己"
```

### 阶段 2：后端抽象

目标：

- 将当前 `backend/agent.py` 抽象成“推理后端接口”

建议方向：

- 保留现有 `transformers` 实现
- 新增 `backend/mlx_agent.py`
- 在统一 `LocalChatAgent` 接口下支持 backend 切换

建议新增的配置项：

```env
MODEL_BACKEND=transformers
```

可选值：

- `transformers`
- `mlx`

### 阶段 3：接入 API 层

目标：

- 让 [backend/api_server.py](/Users/lijingru/code/qwen-2.5/backend/api_server.py) 无需理解底层差异

做法：

- `backend/api_server.py` 只面向统一的 `agent.reply(...)`
- 后端选择由配置决定

这样 Android 端无需修改协议。

## 对当前项目的建议目录

未来如果继续推进 MLX，可考虑逐步变成：

```text
qwen-2.5/
├── backend/
│   ├── agent.py
│   ├── api_server.py
│   ├── transformers_agent.py
│   └── mlx_agent.py
├── config/
│   ├── models/
│   └── backends/
├── experiments/
│   └── mlx_qwen_smoke.py
├── docs/
│   ├── app.md
│   ├── app-v1-plan.md
│   ├── mlx.md
│   └── wechat.md
└── models/
    └── qwen2.5/
```

## 推荐实验顺序

### 1. 先找现成 MLX 模型

优先尝试：

- Qwen 官方提供的 `-MLX` 模型
- 或者 `mlx-community` 下的兼容模型，例如 `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- 对应 `3B` 可先试 `mlx-community/Qwen2.5-3B-Instruct-4bit`

原因：

- 这样可以少做一次模型转换
- 能先验证框架可行性

### 2. 再考虑自行转换

如果目标模型没有现成 MLX 版本，再使用 `mlx_lm.convert`

官方 `mlx-lm` 文档支持转换和量化，例如：

```bash
mlx_lm.convert --model mistralai/Mistral-7B-Instruct-v0.3 -q
```

Qwen 官方文档也给出了类似转换思路：

```bash
mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path mlx/Qwen2.5-7B-Instruct/ -q
```

对本项目当前更实用的版本是：

```bash
mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B-Instruct --mlx-path ./mlx/Qwen2.5-0.5B-Instruct-4bit -q
mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct --mlx-path ./mlx/Qwen2.5-3B-Instruct-4bit -q
```

目前仓库已经实际验证通过的是：

```bash
mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B-Instruct --mlx-path ./mlx/Qwen2.5-0.5B-Instruct-4bit -q
python experiments/mlx_qwen_smoke.py --model ./mlx/Qwen2.5-0.5B-Instruct-4bit --eos-token "<|im_end|>"
```

## 风险

### 风险 1：模型格式差异

影响：

- 不能直接复用现有 `transformers` 权重加载逻辑

应对：

- 把 MLX 作为独立后端实现
- 避免在当前 `backend/agent.py` 里硬塞条件分支

### 风险 2：功能接口差异

影响：

- `mlx-lm` 的 `generate()`、cache、sampling 参数和 `transformers` 并不完全一致

应对：

- 先只对齐最小聊天接口
- 高级参数后续再逐步映射

### 风险 3：模型可用性与转换成本

影响：

- 某些目标模型可能需要转换或额外测试

应对：

- 优先使用已存在的 MLX 模型仓库
- 把自转换留到第二阶段

## 工作量评估

以下评估默认前提是：

- Android 协议不变
- v1.0 仍以 CPU 主线交付
- MLX 先作为 Apple Silicon 可选后端

### 方案 A：只做独立实验

范围：

- 新建 `mlx-lm` 验证脚本
- 选一个 Qwen 模型跑通一轮生成

工作量：

- 约 `0.5 ~ 1` 天

### 方案 B：接入为可切换后端

范围：

- 新增 `mlx_agent.py`
- 抽象后端接口
- API 层无感知切换
- 保留现有 `transformers` 实现

工作量：

- 约 `2 ~ 4` 天

### 方案 C：接入为项目正式能力

范围：

- MLX 后端接入
- 文档、配置、模型注册完善
- Android 端联调与兼容测试
- 回归测试 CPU 与 MLX 两条路线

工作量：

- 约 `4 ~ 7` 天

## 对本项目的建议

最合理的推进顺序是：

1. 继续保持 v1.0 主线使用 CPU
2. 先做 `MLX` 独立 smoke test
3. 验证稳定后，再把 `MLX` 抽象成正式后端

不建议现在立刻把主线从 `transformers` 全量迁到 `mlx-lm`，因为这会把“v1.0 App Demo”与“Apple GPU 实验”绑死在一起。

如果要继续推进成正式的可切换后端，重构计划见 [docs/backend-switch-plan.md](/Users/lijingru/code/qwen-2.5/docs/backend-switch-plan.md)。
实际启动和回归命令见 [docs/backend-runbook.md](/Users/lijingru/code/qwen-2.5/docs/backend-runbook.md)。

## 当前落地状态

这条路线在仓库里的当前状态是：

- 已完成 `CPU 默认 / MPS experimental` 的项目文档整理
- 已新增独立 `MLX` smoke test 脚本
- 已新增单独的 `MLX` 依赖文件
- 已实际验证 `Qwen2.5-0.5B-Instruct` 的本地转换与生成
- 已实际验证 `Qwen2.5-3B-Instruct` 的本地转换与生成
- 还没有把 `mlx-lm` 并入正式后端

这意味着我们现在可以先验证：

- `Qwen2.5-0.5B` 和 `Qwen2.5-3B` 在 `mlx-lm` 上都具备继续工程化接入的基础
- 下一步重点转为 API 层未来是否值得增加 `MODEL_BACKEND=mlx`

## 当前结论

- `CPU default / MPS experimental / MLX follow-up` 是当前最稳的策略
- `mlx-lm` 已经不只是“值得继续实验”，而是“值得进入正式后端重构评估”
- 对本项目来说，`mlx-lm` 的开发工作量是中等，不算特别重，但最好分阶段推进
