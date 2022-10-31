package com.project.chat2learn.common.external.flask.service;

import com.project.chat2learn.common.external.flask.model.response.BaseResponse;
import org.springframework.security.core.Authentication;

public interface BotService {

    BaseResponse messageBotWithId(Long id, String message);
}
