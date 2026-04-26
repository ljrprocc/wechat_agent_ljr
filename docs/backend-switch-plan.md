# 可切换推理后端重构计划

状态：`In Progress`

## 目标

在不破坏现有 Android v1.0 API 协议的前提下，为项目增加可切换的推理后端能力。

目标形态：

- 保留当前 `transformers` 路线
- 新增 `mlx` 路线
- `backend/api_server.py` 不感知底层推理实现差异
- Android 端继续沿用现有 `/api/chat` 协议

## 当前进度

- `M1` 已完成：`runtime.py`、`session_store.py`、`model_registry.py`、`backends/base.py` 已落地
- `M2` 已完成：`transformers_backend.py` 已落地，`agent.py` 已改为依赖 backend 抽象
- 当前 API 已支持 `MODEL_BACKEND=mlx|transformers`
- 当前默认配置仍保持 `MODEL_BACKEND=mlx`

## 当前事实

- `CPU + transformers` 已经是当前 v1.0 主线
- `PyTorch/Transformers + MPS` 在当前 Apple Silicon 机器上不稳定
- `mlx-lm` 已独立验证通过：
  - `Qwen2.5-0.5B-Instruct`
  - `Qwen2.5-3B-Instruct`

因此，重构的核心不是“替换主线”，而是“增加正式的后端切换能力”。

## 重构原则

- 不修改 Android 请求协议
- 不在单个类里堆大量 `if backend == ...`
- 尽量复用现有模型注册、会话存储、接口层
- 先对齐最小聊天能力，再扩展高级参数
- 默认配置按当前 Apple Silicon 实现保持 `mlx`

## 推荐目录结构

建议逐步整理为：

```text
qwen-2.5/
├── backend/
│   ├── __init__.py
│   ├── agent.py
│   ├── api_server.py
│   ├── runtime.py
│   ├── session_store.py
│   ├── model_registry.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── transformers_backend.py
│   │   └── mlx_backend.py
├── config/
│   ├── models/
│   └── backends/
└── docs/
    ├── app.md
    ├── app-v1-plan.md
    ├── backend-switch-plan.md
    └── mlx.md
```

## 范围

### In Scope

- 新增后端抽象接口
- 把 `transformers` 现有实现迁移到独立 backend 模块
- 新增 `mlx` backend 模块
- 新增统一 backend 选择配置
- 复用现有 `SessionStore`
- 保持 `/api/chat` 协议不变
- 增加最小状态输出，便于 `/healthz` 查看当前 backend

### Out of Scope

- 流式输出正式实现
- 多模态能力
- 文生图
- Android 协议改版
- 复杂工具调用
- 统一性能基准平台

## 配置设计

建议新增：

```env
MODEL_BACKEND=mlx
```

可选值：

- `transformers`
- `mlx`

建议保留现有：

- `QWEN_MODEL_CONFIG_PATH`
- `QWEN_DEFAULT_MODEL_ID`
- `AGENT_DB_PATH`
- `AGENT_HISTORY_TURNS`
- `AGENT_SYSTEM_PROMPT`

建议新增 MLX 专属配置：

```env
MLX_QWEN_EOS_TOKEN=<|im_end|>
MLX_QWEN_LOCAL_ROOT=./mlx
MLX_QWEN_PREFER_LOCAL=1
```

## 接口设计

建议抽象一个最小 backend 协议，例如：

```python
class ChatBackend(Protocol):
    def chat(self, messages: list[dict[str, str]], model_id: str) -> str: ...
    def list_models(self) -> list[dict[str, Any]]: ...
    def status(self) -> dict[str, Any]: ...
```

说明：

- `SessionStore`、上下文拼接、`/reset` 等逻辑仍由上层 `LocalChatAgent` 负责
- backend 只负责“给定 messages，返回 answer”
- 这样可以避免把会话逻辑重复写两份

## 里程碑

### M1：抽离公共模块

状态：`Completed`

目标：

- 把当前 `backend/agent.py` 里的共性逻辑拆出来

交付物：

- `runtime.py`
- `session_store.py`
- `model_registry.py`
- `backends/base.py`

验收标准：

- 不改变现有 `/api/chat` 行为
- `transformers` 路线仍能继续工作

### M2：迁移 transformers backend

状态：`Completed`

目标：

- 把当前 `LocalQwenChat` 迁到 `transformers_backend.py`

交付物：

- `backends/transformers_backend.py`
- `agent.py` 改为依赖 backend 抽象

验收标准：

- `MODEL_BACKEND=transformers` 时行为与当前一致
- `/healthz` 可显示当前 backend 类型

### M3：新增 mlx backend

目标：

- 把独立 smoke test 里的可用逻辑收敛成正式 backend

交付物：

- `backends/mlx_backend.py`
- MLX 本地路径优先策略
- `eos_token` 配置支持

验收标准：

- `MODEL_BACKEND=mlx` 时可跑通 `qwen2.5-0.5b`
- `MODEL_BACKEND=mlx` 时可跑通 `qwen2.5-3b`
- Android 端无需修改请求协议

### M4：联调与文档

目标：

- 让后端切换成为真正可使用的项目能力

交付物：

- `.env.example` 更新
- `docs/mlx.md` 更新
- 启动说明与回归清单

验收标准：

- 能分别以 `transformers` / `mlx` 启动 API
- `/healthz` 能正确显示 backend 和模型状态

## 建议实现顺序

1. 先抽公共模块，不先写新功能
2. 再迁移 `transformers` 到独立 backend
3. 之后接入 `mlx`
4. 最后更新健康检查与文档

## 风险与缓解

### 风险 1：两个 backend 的采样参数不完全一致

影响：

- 同样的温度和 top-p 不一定返回同风格结果

缓解：

- v1 只对齐“可用性”，不承诺逐 token 行为一致

### 风险 2：MLX 模型来源不一致

影响：

- 社区仓库与本地转换目录表现可能不同

缓解：

- 优先使用本地转换目录
- 把 `mlx-community` 仅作为兜底来源

### 风险 3：代码拆分期间引入回归

影响：

- 现有 CPU 主线可能被破坏

缓解：

- 先迁移 `transformers`，保证旧链路先通过
- 每个里程碑后都做一次最小回归

## 工作量评估

基于当前仓库状态，重构可切换后端的工作量大致如下：

- M1 + M2：约 `1 ~ 2` 天
- M3：约 `1 ~ 2` 天
- M4：约 `0.5 ~ 1` 天

合计：

- 约 `2.5 ~ 5` 天

如果只要求“本地可跑、最小文档齐全”，更接近 `2.5 ~ 3.5` 天。

如果要连同 Android 端回归、更多模型状态展示一起做完，更接近 `4 ~ 5` 天。

## 建议结论

当前最合理的推进方式是：

1. 保持 `CPU + transformers` 作为 v1.0 主线
2. 把 `mlx` 作为正式候选后端开始重构
3. 先完成“可切换后端”能力，再决定是否把默认 Apple Silicon 路线切到 `mlx`
