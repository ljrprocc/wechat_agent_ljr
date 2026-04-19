# 微信小程序调用本地部署模型

## 目标

通过微信小程序调用当前仓库里的本地 Qwen 模型服务。

推荐链路：

`微信小程序 -> HTTPS API -> backend/agent.py -> 本地/缓存的 Qwen 权重`

这里的关键点不是模型本身，而是“小程序网络访问规则”。

## 先说结论

小程序不适合直接访问开发机上的 `localhost`。

更稳的做法是：

1. 先把模型封装成一个标准 HTTP API。
2. 再把这个 API 暴露成一个可访问的 HTTPS 域名。
3. 最后在小程序后台把该域名加入合法域名列表。

## 平台限制

根据腾讯云官方的微信小程序网络说明：

- 小程序需要预先配置通信域名。
- `wx.request` 走 HTTPS。
- `wx.connectSocket` 走 WSS。
- 合法域名不能直接使用 IP 地址或 `localhost`。
- 默认和最大超时时间都是 60 秒。

这意味着：

- 当前仓库如果只在本地 `127.0.0.1:8000` 启动，正式小程序是不能直接访问的。
- 如果模型回复时间接近 60 秒，普通 `wx.request` 很容易超时。

## 推荐项目形态

和 Android 路线一样，推荐先统一出一个通用后端接口：

- `GET /healthz`
- `POST /api/chat`

请求体示例：

```json
{
  "session_id": "wx-openid-001",
  "message": "你好"
}
```

响应体示例：

```json
{
  "reply": "你好，我可以为你提供帮助。",
  "session_id": "wx-openid-001"
}
```

## 最小可行部署方式

### 方案 A：本机启动服务 + HTTPS 隧道

这是一条最适合做 demo 的微信路线。

1. 本地启动 FastAPI 模型服务。
2. 用 `cloudflared`、`ngrok` 或 `frp` 把本地服务暴露成一个公网 HTTPS 域名。
3. 在微信小程序后台配置该域名为 `request` 合法域名。
4. 小程序前端通过 `wx.request` 调用这个 HTTPS 地址。

链路如下：

`微信小程序 -> https://your-domain/api/chat -> 本机 FastAPI -> 本地模型`

### 方案 B：把模型服务部署到云服务器

适合后续稳定演示或准备上线。

推荐形态：

- Linux 服务器
- `Nginx + HTTPS`
- FastAPI 模型服务
- 域名已配置证书

## 小程序前端最小调用方式

示例：

```javascript
wx.request({
  url: 'https://api.example.com/api/chat',
  method: 'POST',
  data: {
    session_id: 'wx-demo-user',
    message: '你好'
  },
  success(res) {
    console.log(res.data.reply)
  }
})
```

如果后面要做流式输出，再把接口升级到 `wx.connectSocket`。

## 登录与会话

如果你只是先做最小 demo，`session_id` 可以先由前端临时生成。

如果要更正式一点，建议：

1. 小程序前端调用 `wx.login()`
2. 把 `code` 发到你的后端
3. 后端再换取 `openid`
4. 用 `openid` 作为 `session_id`

这样会话身份更稳定。

## 当前仓库如何接入

当前仓库已经有：

- [backend/agent.py](/Users/lijingru/code/qwen-2.5/backend/agent.py)：本地模型加载、SQLite 记忆、对话封装
- [channels/wechat/app.py](/Users/lijingru/code/qwen-2.5/channels/wechat/app.py)：公众号/企业微信回调入口

对于小程序，不建议复用微信回调接口；更合理的是新增一个通用 API 入口，例如 `backend/api_server.py`。

建议的最小实现：

```python
from fastapi import FastAPI
from pydantic import BaseModel

from backend.agent import build_agent_from_env

app = FastAPI()
agent = build_agent_from_env()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    reply = agent.reply(req.session_id, req.message)
    return {"reply": reply, "session_id": req.session_id}
```

启动命令：

```bash
uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

## 微信路线的额外难点

- 需要 HTTPS 域名。
- 需要在小程序后台配置合法域名。
- 不能直接把 `localhost` 或裸 IP 当成正式通信地址。
- 请求超时上限是 60 秒，对首轮模型加载不友好。
- 如果后续接登录、用户体系、审核发布，工程成本会继续增加。

## 适合什么时候做

如果你的目标是“入口在微信里，用户不用安装 App”，小程序路线是对的。

如果你的目标是“先把模型服务和客户端联调跑起来”，小程序通常不是第一优先级。

## 推荐结论

小程序适合做第二阶段。

建议先把 Android 或通用 Web 客户端跑通，再迁移到小程序。

## 参考资料

- 微信小程序网络说明：
  https://intl.cloud.tencent.com/document/product/1219/61745
- `wx.request` 接口说明：
  https://intl.cloud.tencent.com/document/product/1219/57750
- 小程序开发管理与域名配置：
  https://intl.cloud.tencent.com/document/product/1219/60624
