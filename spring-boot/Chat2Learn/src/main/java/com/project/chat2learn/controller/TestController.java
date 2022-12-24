package com.project.chat2learn.controller;

import com.project.chat2learn.common.external.flask.service.impl.BotServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class TestController {

    private BotServiceImpl modelService;

    @Autowired
    public TestController(BotServiceImpl modelService) {
        this.modelService = modelService;
    }

}
