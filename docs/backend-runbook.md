# Backend 运行手册

## 目标

为当前项目提供一份可直接执行的后端运行与回归手册，覆盖：

- `mlx` 启动
- `transformers` 启动
- `/healthz` 检查
- `/api/chat` 调试请求
- 最小回归清单

## 前置条件

默认假设：

- 已创建并激活 `conda` 环境 `app`
- 已安装：
  - `pip install -r requirements.txt`
  - `pip install -r requirements-mlx.txt`
- 项目根目录是：
  - `/Users/lijingru/code/qwen-2.5`

推荐先执行：

```bash
conda activate app
cd /Users/lijingru/code/qwen-2.5
```

## 环境加载

如果你使用 `.env`，推荐每次启动前都显式加载：

```bash
set -a
source .env
set +a
```

## 启动方式

### 1. 启动 MLX backend

```bash
conda activate app
cd /Users/lijingru/code/qwen-2.5

set -a
source .env
set +a

MODEL_BACKEND=mlx uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

说明：

- 当前默认推荐就是 `MODEL_BACKEND=mlx`
- 如果 `.env` 里已经写了 `MODEL_BACKEND=mlx`，命令行可以不重复写

### 2. 启动 transformers backend

```bash
conda activate app
cd /Users/lijingru/code/qwen-2.5

set -a
source .env
set +a

MODEL_BACKEND=transformers QWEN_DEVICE=cpu uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

说明：

- 当前 Apple Silicon 上不建议把 API 主线跑在 `MPS`
- `transformers` 回归建议统一先用 `cpu`

## 健康检查

服务启动后，先检查：

```bash
curl http://127.0.0.1:8000/healthz
```

期望关注这些字段：

- `agent.model_backend`
- `agent.default_model_id`
- `agent.supported_models`
- `agent.loaded_models`

`mlx` 模式下还会看到：

- `agent.mlx_local_root`
- `agent.mlx_prefer_local`
- `agent.model_sources`

`transformers` 模式下还会看到：

- `agent.device`
- `agent.device_map`
- `agent.torch_dtype`
- `agent.attn_implementation`

## 聊天请求示例

### 1. 普通请求

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-user",
    "message": "你好，请用一句话介绍自己。",
    "model_id": "qwen2.5-0.5b",
    "stream": false
  }'
```

### 2. 开启 debug

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-user",
    "message": "你好，请用一句话介绍自己。",
    "model_id": "qwen2.5-0.5b",
    "stream": false,
    "debug": true
  }'
```

期望 `debug` 中至少包含：

- `backend`
- `latency_ms`
- `generation_ms`
- `prompt_tokens`
- `generated_tokens`
- `tokens_per_second`

## 最小回归清单

### MLX 回归

1. `MODEL_BACKEND=mlx` 启动服务成功
2. `GET /healthz` 返回 `agent.model_backend = "mlx"`
3. `POST /api/chat` 使用 `qwen2.5-0.5b` 返回正常文本
4. `POST /api/chat` 使用 `qwen2.5-3b` 返回正常文本
5. `debug=true` 时返回吞吐量和延迟

### transformers 回归

1. `MODEL_BACKEND=transformers QWEN_DEVICE=cpu` 启动服务成功
2. `GET /healthz` 返回 `agent.model_backend = "transformers"`
3. `POST /api/chat` 使用 `qwen2.5-0.5b` 返回正常文本
4. `debug=true` 时返回吞吐量和延迟

## 本地快速回归命令

### MLX 0.5B

```bash
conda run --no-capture-output -n app python -c "from backend.agent import build_agent_from_env; agent=build_agent_from_env(); result=agent.reply_with_debug('runbook-mlx-05','你好，请用一句话介绍自己。','qwen2.5-0.5b', debug=True); print(result)"
```

### MLX 3B

```bash
conda run --no-capture-output -n app python -c "from backend.agent import build_agent_from_env; agent=build_agent_from_env(); result=agent.reply_with_debug('runbook-mlx-3b','你好，请用一句话介绍自己。','qwen2.5-3b', debug=True); print(result)"
```

### transformers CPU

```bash
MODEL_BACKEND=transformers QWEN_DEVICE=cpu QWEN_MAX_NEW_TOKENS=16 conda run --no-capture-output -n app python -c "from backend.agent import build_agent_from_env; agent=build_agent_from_env(); result=agent.reply_with_debug('runbook-tf','你好，请用一句话介绍自己。','qwen2.5-0.5b', debug=True); print(result)"
```

## 常见问题

### 1. `/healthz` 正常，但第一次回复很慢

原因：

- 模型是懒加载的
- 第一次对话会触发真实加载

建议：

- 先跑一次最小请求做预热

### 2. transformers 模式下出现 Hugging Face 连接报错

原因：

- tokenizer 或模型元信息可能需要联网校验

建议：

- 确认网络可用
- 或者提前完成缓存预热

### 3. MLX 模式下模型源不对

排查：

- 看 `/healthz` 里的 `agent.model_sources`
- 确认是否命中了本地 `./mlx/...` 目录
