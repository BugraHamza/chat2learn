package com.project.chat2learn.common.external.flask.service;

import com.project.chat2learn.common.external.flask.model.response.ChatBotResponse;
import com.project.chat2learn.common.external.flask.model.response.GrammerCheckResponse;

import java.util.concurrent.CompletableFuture;

public interface BotService {

    CompletableFuture<GrammerCheckResponse> checkGrammer(String message);

    CompletableFuture<ChatBotResponse> messageBot(Long id,String message);
}
