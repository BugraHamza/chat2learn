package com.project.chat2learn.controller;

import com.project.chat2learn.common.external.flask.client.FlaskFeignClient;
import com.project.chat2learn.common.external.flask.service.impl.ModelServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class TestController {

    private ModelServiceImpl modelService;

    @Autowired
    public TestController(ModelServiceImpl modelService) {
        this.modelService = modelService;
    }

    @GetMapping(path = "/{id}")
    public ResponseEntity<String> getModel(@PathVariable Long id) {
        return new ResponseEntity<>(modelService.getString(id), HttpStatus.OK);
    }
}
