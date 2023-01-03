package com.project.chat2learn.common.external.flask.service.impl;

import com.project.chat2learn.common.external.flask.client.FlaskFeignClient;
import com.project.chat2learn.common.external.flask.model.request.BaseRequest;
import com.project.chat2learn.common.external.flask.model.response.ChatBotResponse;
import com.project.chat2learn.common.external.flask.model.response.GrammerCheckResponse;
import com.project.chat2learn.common.external.flask.service.BotService;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

@Service
@Log4j2
public class BotServiceImpl implements BotService {

    private final FlaskFeignClient client;
    @Autowired
    public BotServiceImpl(FlaskFeignClient client) {
        this.client = client;
    }

    @Override
    @Async
    public CompletableFuture<GrammerCheckResponse> checkGrammer(String message) {
        BaseRequest request = new BaseRequest();
        request.setMessage(message);
        return CompletableFuture.completedFuture(client.check(request));
    }

    @Override
    @Async
    public CompletableFuture<ChatBotResponse> messageBot(Long id, String message) {
        BaseRequest request = new BaseRequest();
        request.setMessage(message);
        return CompletableFuture.completedFuture(client.chat(id,request));
    }
}
