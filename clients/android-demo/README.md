# Android Demo

## 目标

这是 APP v1.0 的最小 Android Demo。

当前版本用于验证下面这条链路：

`Android App -> backend/api_server.py -> backend/agent.py -> Qwen2.5`

## 默认约定

- 默认后端地址：`http://10.0.2.2:8000/`
- 默认模型：`qwen2.5-0.5b`
- 访问方式：局域网或 Android 模拟器
- `stream` 字段已预留，但当前固定为非流式

## 启动后端

在仓库根目录执行：

```bash
uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

## 打开 Android 工程

使用 Android Studio 打开：

`clients/android-demo/`

首次打开后等待 Gradle 同步完成。

## 联调说明

- Android 模拟器访问宿主机后端时，保持默认地址 `http://10.0.2.2:8000/`
- 如果使用真机局域网调试，把后端地址改成电脑的局域网 IP，例如 `http://192.168.1.10:8000/`
- 当前 Demo 已启用明文 HTTP，便于局域网联调

## 当前界面

- 单页聊天界面
- 后端地址输入框
- 模型选择下拉框
- 对话管理入口占位

## 当前限制

- 未实现流式输出
- 未实现真正的对话管理
- 未做公网访问
- 未包含 Gradle Wrapper，需要在 Android Studio 中打开或自行生成 Wrapper
