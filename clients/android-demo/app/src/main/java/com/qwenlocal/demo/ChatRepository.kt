package com.qwenlocal.demo

class ChatRepository {
    suspend fun sendMessage(
        baseUrl: String,
        request: ChatRequest,
    ): ChatResponse {
        return ChatApiFactory.create(baseUrl).chat(request)
    }
}
