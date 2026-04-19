package com.qwenlocal.demo

data class ModelOption(
    val id: String,
    val label: String,
)

data class ChatMessage(
    val role: MessageRole,
    val content: String,
)

enum class MessageRole {
    USER,
    ASSISTANT,
}

data class ChatRequest(
    val session_id: String,
    val message: String,
    val model_id: String,
    val stream: Boolean = false,
)

data class ChatResponse(
    val reply: String,
    val session_id: String,
    val model_id: String,
    val stream: Boolean,
)
