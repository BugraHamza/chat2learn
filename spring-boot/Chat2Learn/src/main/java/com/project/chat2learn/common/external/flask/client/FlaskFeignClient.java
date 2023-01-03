package com.project.chat2learn.common.external.flask.client;

import com.project.chat2learn.common.external.flask.model.request.BaseRequest;
import com.project.chat2learn.common.external.flask.model.response.ChatBotResponse;
import com.project.chat2learn.common.external.flask.model.response.GrammerCheckResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.*;

@FeignClient(name = "modelClient",contextId = "modelClient", url = "http://localhost:9090")
public interface FlaskFeignClient {

    @PostMapping(path = "/chat/{id}")
    ChatBotResponse chat(@PathVariable Long id, @RequestBody BaseRequest request);

    @PostMapping(path = "/check")
    GrammerCheckResponse check(@RequestBody BaseRequest request);
}
