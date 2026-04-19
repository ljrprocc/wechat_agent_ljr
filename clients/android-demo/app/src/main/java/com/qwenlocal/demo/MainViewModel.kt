package com.qwenlocal.demo

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import java.util.UUID

class MainViewModel : ViewModel() {
    private val repository = ChatRepository()
    private val sessionId = UUID.randomUUID().toString()

    val messages = mutableStateListOf<ChatMessage>()
    val models = listOf(
        ModelOption(id = "qwen2.5-0.5b", label = "Qwen2.5 0.5B"),
        ModelOption(id = "qwen2.5-3b", label = "Qwen2.5 3B"),
    )

    var backendUrl by mutableStateOf(BuildConfig.DEFAULT_BASE_URL)
        private set

    var selectedModelId by mutableStateOf(BuildConfig.DEFAULT_MODEL_ID)
        private set

    var currentInput by mutableStateOf("")
        private set

    var statusText by mutableStateOf("局域网模式：默认连接 Android 模拟器的 http://10.0.2.2:8000/")
        private set

    var isSending by mutableStateOf(false)
        private set

    fun updateBackendUrl(value: String) {
        backendUrl = value
    }

    fun updateSelectedModel(modelId: String) {
        selectedModelId = modelId
        statusText = "当前模型：$modelId"
    }

    fun updateInput(value: String) {
        currentInput = value
    }

    fun sendMessage() {
        val message = currentInput.trim()
        val baseUrl = backendUrl.trim()
        if (message.isEmpty() || baseUrl.isEmpty() || isSending) {
            return
        }

        messages += ChatMessage(role = MessageRole.USER, content = message)
        currentInput = ""
        isSending = true
        statusText = "请求中..."

        viewModelScope.launch {
            runCatching {
                repository.sendMessage(
                    baseUrl = baseUrl,
                    request = ChatRequest(
                        session_id = sessionId,
                        message = message,
                        model_id = selectedModelId,
                        stream = false,
                    ),
                )
            }.onSuccess { response ->
                messages += ChatMessage(role = MessageRole.ASSISTANT, content = response.reply)
                statusText = "已连接 ${response.model_id}"
            }.onFailure { throwable ->
                statusText = "请求失败：${throwable.message ?: "未知错误"}"
            }

            isSending = false
        }
    }
}
