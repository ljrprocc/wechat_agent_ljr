# Android App 远程调用本地模型

## 目标

将当前仓库里的本地 Qwen 模型封装成一个 HTTP API，再由 Android App 远程调用。

推荐链路：

`Android App -> FastAPI API -> backend/agent.py -> 本地/缓存的 Qwen 权重`

这条路线的好处是：

- Android 端只需要会发 HTTP 请求，不需要直接处理模型加载。
- 后端可以继续复用当前仓库里的 [backend/agent.py](/Users/lijingru/code/qwen-2.5/backend/agent.py)。
- 同一个 API 后面还可以继续给 Web、桌面端、小程序复用。

## 当前目录建议

当前仓库已经整理为以下方向：

```text
qwen-2.5/
├── backend/
├── channels/
├── docs/
├── models/
│   └── qwen2.5/
├── .env.example
└── requirements.txt
```

其中：

- Qwen2.5 研究脚本位于 [models/qwen2.5/qwen.py](/Users/lijingru/code/qwen-2.5/models/qwen2.5/qwen.py) 和 [models/qwen2.5/qwen_official.py](/Users/lijingru/code/qwen-2.5/models/qwen2.5/qwen_official.py)
- 通用对话服务位于 [backend/agent.py](/Users/lijingru/code/qwen-2.5/backend/agent.py)
- 微信通道入口位于 [channels/wechat/app.py](/Users/lijingru/code/qwen-2.5/channels/wechat/app.py)

## 后端策略

当前项目的推荐后端策略是：

- v1.0 默认使用 `CPU`
- `PyTorch/Transformers + MPS` 仅作为实验路径
- Apple Silicon 上如果要继续追本地 GPU 推理，优先研究 `MLX / mlx-lm`

当前代码实现补充：

- `backend/agent.py` 已经开始按 `MLX` 后端重构
- 当前 API 侧模型调用固定走 `MLX`
- `CPU + transformers` 仍保留为前一阶段的规划基线，不再代表当前实现状态

补充说明：

- 我们已经验证过 `CPU` 可以稳定跑通 v1.0 链路
- `PyTorch/Transformers + MPS` 在当前机器上出现了 `MPSTemporaryNDArray ... total bytes of NDArray > 2**32`
- 因此，MPS 不再作为 v1.0 默认方案

MLX 路线规划见 [docs/mlx.md](/Users/lijingru/code/qwen-2.5/docs/mlx.md)。
当前独立实验脚本见 [experiments/mlx_qwen_smoke.py](/Users/lijingru/code/qwen-2.5/experiments/mlx_qwen_smoke.py)。

## 版本路线

### APP v1.0：基础 Demo

目标：

- 跑通 Android App 调用本地部署模型的最小闭环
- 先验证“客户端 -> API -> 本地模型”的可行性

范围：

- 单聊天页面
- 单个 `/api/chat` 接口
- 局域网/模拟器访问
- 仅支持 `Qwen2.5-0.5B` 与 `Qwen2.5-3B`
- 保留 `model_id` 和 `stream` 扩展接口，但不实现流式输出
- 默认后端为 `CPU`

详细计划见 [docs/app-v1-plan.md](/Users/lijingru/code/qwen-2.5/docs/app-v1-plan.md)。

### APP v1.x：功能迭代

目标：

- 在 v1.0 Demo 跑通后，逐步完善聊天页面体验

当前已记录的方向：

- 聊天页面优化
- 对话管理
- 不改变 v1.0 已经固定的基础协议
- 评估是否引入 `MLX` 作为 Apple Silicon 上的可选推理后端

### APP v2.0：多模型支持

目标：

- 在统一 API 下支持多种语言模型家族

建议能力：

- 模型注册表
- 模型切换配置
- 每个模型独立参数模板
- App 内模型选择入口

说明：

- v1.x 里的 `Qwen2.5-0.5B` / `Qwen2.5-3B` 更像是同一模型家族下的两种规格
- 真正的多模型选择能力放到 v2.x 再扩展

### APP v3.0：文生图

目标：

- 在文本对话之外支持文生图能力

建议能力：

- 图像生成任务接口
- 图片生成状态轮询或流式反馈
- 图片缓存与历史记录
- 模型能力隔离，避免文本接口和图像接口耦合

## 推荐项目形态

建议在当前仓库内统一一个通用接口层，而不是让客户端直接调用 [channels/wechat/app.py](/Users/lijingru/code/qwen-2.5/channels/wechat/app.py) 里的微信回调入口。

推荐新增的最小接口：

- `GET /healthz`
- `POST /api/chat`

推荐请求体：

```json
{
  "session_id": "android-user-001",
  "message": "你好，帮我介绍一下自己",
  "model_id": "qwen2.5-0.5b",
  "stream": false
}
```

推荐响应体：

```json
{
  "reply": "你好，我是一个本地部署的 Qwen 助手。",
  "session_id": "android-user-001",
  "model_id": "qwen2.5-0.5b",
  "stream": false
}
```

## 服务端部署方案

### 方案 A：本机开发联调

适合先做最小 demo。

1. 在本机启动 FastAPI 服务。
2. Android 模拟器使用 `10.0.2.2` 访问宿主机。
3. 真机与电脑处于同一局域网时，使用电脑局域网 IP 访问。

说明：

- Android 官方文档说明，模拟器里的 `10.0.2.2` 会映射到开发机的 `127.0.0.1`。
- 如果你用的是 `http://`，Android 9 及以上默认不允许明文流量，需要显式开启。
- 当前路线明确以局域网访问为主，暂不纳入公网部署。

## Android 端最小要求

### 1. 开启网络权限

`AndroidManifest.xml`：

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

### 2. 开发阶段允许 HTTP

如果后端暂时没有 HTTPS，可以在开发阶段先允许明文流量。

`AndroidManifest.xml`：

```xml
<application
    android:usesCleartextTraffic="true" />
```

更稳的方式是配置 `network_security_config.xml`，只对白名单域名或内网地址放开 HTTP。

### 3. App 内调用接口

最小请求逻辑可以非常简单，用 `OkHttp`、`Retrofit` 或 `Ktor` 都行。下面是伪代码：

```kotlin
POST http://10.0.2.2:8000/api/chat
{
  "session_id": "android-user-001",
  "message": "你好",
  "model_id": "qwen2.5-0.5b",
  "stream": false
}
```

## 从本地开发到远程访问

说明：

- 以下内容仅作为后续扩展参考，不属于当前确认的 v1.0 范围
- 当前 v1.0 仍然只以局域网/模拟器访问为准

### 方案 B：通过隧道工具暴露 HTTPS

适合“手机不在同一局域网”或“想让别人临时体验”。

可选方式：

- `cloudflared tunnel`
- `ngrok`
- `frp`

链路会变成：

`Android App -> HTTPS 公网域名 -> 你的本机 FastAPI -> 本地模型`

这时 Android 端就不需要额外放开 HTTP 了。

### 方案 C：正式部署到云服务器

适合长期可用。

推荐形态：

- 模型服务跑在一台 Linux 服务器上
- 前面用 `Nginx + HTTPS`
- Android 只请求正式域名，例如 `https://api.example.com/api/chat`

## 建议的最小后端接口

为了同时服务 Android 和后续的小程序，推荐先补这样一个文件，例如 `backend/api_server.py`：

```python
from fastapi import FastAPI
from pydantic import BaseModel

from backend.agent import build_agent_from_env

app = FastAPI()
agent = build_agent_from_env()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model_id: str = "qwen2.5-0.5b"
    stream: bool = False


@app.get("/healthz")
def healthz():
    return {"ok": True, "agent": agent.status()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    reply = agent.reply(req.session_id, req.message)
    return {
        "reply": reply,
        "session_id": req.session_id,
        "model_id": req.model_id,
        "stream": False,
    }
```

启动命令：

```bash
uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

默认环境建议：

- `QWEN_DEVICE=cpu`
- 先在 CPU 上把 v1.0 功能链路跑通
- Apple GPU 路线作为单独实验项推进，不与 v1.0 主线耦合

## Android 路线的优点

- 对后端约束最少。
- 可以直接走局域网或模拟器调试。
- 不依赖微信平台审核、域名白名单和小程序后台配置。
- 最小 demo 成本最低。

## 这条路线的主要风险

- 如果模型首轮加载较慢，App 首次请求会卡住较久。
- 如果本机没有公网入口，外网设备无法访问。
- 如果接口长时间不返回，App 端需要自己做超时和重试。
- 两个 Qwen2.5 规格的性能差异会带来体验差异。
- `PyTorch/Transformers + MPS` 在 Apple Silicon 上可能不稳定。

## 推荐结论

如果目标是“先尽快跑通一个最小可演示版本”，优先做 Android App。

当前建议是：

- 按 [docs/app-v1-plan.md](/Users/lijingru/code/qwen-2.5/docs/app-v1-plan.md) 推进 v1.0
- 当前优先补齐 `backend/api_server.py` 和 Android Demo
- 复杂的聊天页面能力留到 `v1.x` 继续迭代

## 参考资料

- Android 网络安全配置：
  https://developer.android.com/privacy-and-security/security-config
- Android 模拟器网络地址 `10.0.2.2`：
  https://developer.android.com/studio/run/emulator-networking-address
