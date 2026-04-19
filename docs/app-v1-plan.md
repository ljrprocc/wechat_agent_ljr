# APP v1.0 开发计划

状态：`In Progress`

说明：

- 本文档用于指导 v1.0 的实际落地。
- 当前目标是先做一个最小 Android Demo，验证“移动端远程调用本地部署模型”这条链路。

## 已确认决策

- v1.0 引入独立服务端入口，新增 `backend/api_server.py`
- Android 端先做单页聊天界面
- 后续 `v1.x` 再继续讨论聊天页面和对话管理的功能设计
- v1.0 不做流式输出，但保留流式输出接口
- `v1.x` 暂时只支持 `Qwen2.5-0.5B` 和 `Qwen2.5-3B`
- 后端必须保留模型选择接口，为 `v2.x` 的多模型能力做准备
- v1.0 访问范围仅限局域网和模拟器，不考虑公网部署

## v1.0 目标

交付一个可演示的 Android 最小版本，完成以下闭环：

`Android Demo App -> FastAPI /api/chat -> backend/agent.py -> 本地 Qwen 模型`

用户可以在 Android 端输入一条消息，后端返回模型回复，并保留最基础的多轮上下文。

同时，v1.0 需要提前固定两类“预留能力接口”：

- 预留 `model_id`，但仅允许 Qwen2.5 的两个规格
- 预留 `stream`，但 v1.0 统一按非流式响应处理

## v1.0 范围

### In Scope

- 新增一个通用 API 服务入口，例如 `backend/api_server.py`
- 提供 `GET /healthz`
- 提供 `POST /api/chat`
- 复用 `backend/agent.py` 作为模型对话服务
- 新增一个最小 Android Demo 工程
- Android 端提供单页聊天界面
- 支持配置后端地址
- 支持显示请求中、成功和失败状态
- 支持在接口层传入 `model_id`
- 在接口协议中预留 `stream` 字段
- v1.0 仅支持局域网访问
- v1.0 支持 `Qwen2.5-0.5B` 与 `Qwen2.5-3B`

### Out of Scope

- 用户登录与鉴权
- 实际流式输出能力
- 语音输入
- 长会话管理与摘要
- 多模型家族切换
- 文生图
- iOS 客户端
- 微信小程序正式接入
- 公网访问与外网部署

## 推荐目录结构

当前建议逐步整理为：

```text
qwen-2.5/
├── backend/
│   ├── __init__.py
│   ├── agent.py
│   └── api_server.py           # v1.0 计划新增
├── channels/
│   ├── __init__.py
│   └── wechat/
│       ├── __init__.py
│       └── app.py
├── clients/
│   └── android-demo/           # v1.0 计划新增
├── docs/
│   ├── app.md
│   ├── app-v1-plan.md
│   └── wechat.md
├── config/
│   └── models/                 # v1.0 计划新增
├── models/
│   └── qwen2.5/
│       ├── qwen.py
│       └── qwen_official.py
├── .env.example
├── requirements.txt
├── wechat_agent.py             # 兼容入口
└── wechat_app.py               # 兼容入口
```

## 里程碑

### M1：后端 API 统一

目标：

- 把“聊天服务”和“渠道回调”解耦
- 提供可供 Android、小程序、Web 复用的统一 API

交付物：

- `backend/api_server.py`
- API 请求/响应模型定义
- 模型选择接口定义
- 流式输出预留字段定义
- 基础错误处理
- 本地启动说明

验收标准：

- `GET /healthz` 能返回服务状态
- `POST /api/chat` 能返回模型文本
- 使用同一 `session_id` 时可保留多轮上下文
- `model_id` 仅允许两个 Qwen2.5 规格
- `stream` 字段存在但在 v1.0 固定返回非流式结果
- 服务端默认面向局域网启动与联调

### M2：Android Demo 客户端

目标：

- 做一个最小但可演示的 Android 客户端

交付物：

- `clients/android-demo/`
- 单页聊天 UI
- 基础网络请求层
- 错误提示和加载状态
- 模型切换入口占位
- 对话列表/对话管理入口占位，先不实现实际功能

验收标准：

- App 能连到本机或局域网内的后端
- 用户发出消息后可看到模型回复
- 接口失败时界面能提示错误
- UI 保留后续扩展对话管理的位置

### M3：基础可演示性提升

目标：

- 让 demo 更适合演示，而不是只在开发机上可用

交付物：

- 环境配置说明
- 局域网访问说明
- 模型预热建议
- 最小日志方案

验收标准：

- 新人按照文档可以本地启动服务
- Android 模拟器可以打通一次完整对话
- 至少覆盖一次首轮加载场景
- 局域网内真机或模拟器至少一种方式可稳定联调

## 建议实现顺序

1. 先补 `backend/api_server.py`
2. 固定 `POST /api/chat` 的请求响应协议
3. 再本机自测 API
4. 再创建 `clients/android-demo/`
5. 最后调通 Android 到本地服务

## 建议接口草案

推荐把 v1.0 的协议先固定成下面这样：

请求体：

```json
{
  "session_id": "android-user-001",
  "message": "你好",
  "model_id": "qwen2.5-0.5b",
  "stream": false
}
```

响应体：

```json
{
  "reply": "你好，我是一个本地部署的 Qwen 助手。",
  "session_id": "android-user-001",
  "model_id": "qwen2.5-0.5b",
  "stream": false
}
```

字段说明：

- `model_id`：v1.0 仅允许 `qwen2.5-0.5b` 和 `qwen2.5-3b`
- `stream`：v1.0 必须传 `false`，或由服务端默认补为 `false`

## 技术决策建议

### 服务端

- 使用 `FastAPI`
- 沿用当前 `.env` 配置方式
- 延迟加载模型，但在文档里提供手动预热命令
- 增加模型注册配置，哪怕 v1.0 只先支持两个 Qwen2.5 规格
- `api_server` 与渠道回调入口彻底解耦

### Android 端

- UI：Jetpack Compose
- 网络：Retrofit 或 OkHttp
- 最小状态管理：ViewModel
- 单页聊天布局，但预留“对话管理”扩展位
- 预留模型选择入口，但 v1.0 可先做最小可见控件或配置入口

说明：

- 这是“最稳妥的建议”，不是强制框架锁定。
- 如果你更偏向原生 XML 布局，也可以在评审后调整。

## 风险与缓解

### 风险 1：模型首次加载过慢

影响：

- 首次请求耗时长，移动端体验差

缓解：

- 在文档中加入预热步骤
- v1.0 先用较小模型

### 风险 2：局域网访问不稳定

影响：

- 真机联调复杂

缓解：

- 先以 Android 模拟器为主
- 明确 v1.0 仅支持局域网
- 真机联调作为补充验证，不作为首要验收路径

### 风险 3：接口定义频繁变化

影响：

- 客户端和服务端反复对齐

缓解：

- 在 v1.0 内固定 `POST /api/chat` 的最小协议
- 提前固定 `model_id` 和 `stream` 两个扩展字段

### 风险 4：双模型支持导致配置分叉

影响：

- `0.5B` 和 `3B` 的推理性能差异可能影响体验

缓解：

- v1.0 优先默认 `qwen2.5-0.5b`
- `qwen2.5-3b` 作为可选模型保留
- 用模型注册配置统一管理两者入口

## v1.0 评审重点

建议你审阅时重点看这几项：

- `POST /api/chat` 是否固定为当前协议
- `model_id` 的保留方式是否合适
- `stream` 字段只保留不实现的策略是否合适
- 单页聊天界面是否已经足够支撑 v1.0 Demo
- “对话管理”是否放入 `v1.x` 再讨论实现

## 进入开发前的决策

在开始执行 v1.0 之前，建议确认：

1. Android 客户端是否采用 Jetpack Compose
2. v1.0 模型映射命名是否采用 `qwen2.5-0.5b` / `qwen2.5-3b`
3. 模型选择入口在 Android 端是先显示控件还是先隐藏在设置页

## 结论

v1.0 的目标应保持克制：

- 先证明“本地模型服务可被 Android 客户端稳定调用”
- 不要在 v1.0 同时追求真正的多模型家族切换、流式输出和图像生成
- 但接口层要为 `v1.x` 和 `v2.x` 的扩展留好位置

这份计划完成后，下一步应该是“审阅并确认范围”，而不是直接开始编码。
