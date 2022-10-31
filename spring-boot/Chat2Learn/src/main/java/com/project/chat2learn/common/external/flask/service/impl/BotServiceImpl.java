package com.project.chat2learn.common.external.flask.service.impl;

import com.project.chat2learn.common.external.flask.client.FlaskFeignClient;
import com.project.chat2learn.common.external.flask.model.request.BaseRequest;
import com.project.chat2learn.common.external.flask.model.response.BaseResponse;
import com.project.chat2learn.common.external.flask.service.BotService;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
@Log4j2
public class BotServiceImpl implements BotService {

    private final FlaskFeignClient client;
    @Autowired
    public BotServiceImpl(FlaskFeignClient client) {
        this.client = client;
    }

    @Override
    public BaseResponse messageBotWithId(Long id, String message) {
        BaseRequest request = new BaseRequest();
        request.setMessage(message);
        return client.chat(id,request);
    }
}
