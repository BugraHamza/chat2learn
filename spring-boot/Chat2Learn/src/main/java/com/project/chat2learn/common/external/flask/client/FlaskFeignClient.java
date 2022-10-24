package com.project.chat2learn.common.external.flask.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

@FeignClient(name = "modelClient",contextId = "modelClient", url = "http://localhost:9090")
public interface FlaskFeignClient {

    @GetMapping(path = "/{id}")
    String getString(@PathVariable Long id);
}
