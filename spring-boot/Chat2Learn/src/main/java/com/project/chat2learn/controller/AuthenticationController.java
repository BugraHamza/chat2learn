package com.project.chat2learn.controller;

import com.project.chat2learn.service.AuthenticationService;
import com.project.chat2learn.service.model.request.LoginRequest;
import com.project.chat2learn.service.model.request.RegisterRequest;
import com.project.chat2learn.service.model.response.LoginResponse;
import com.project.chat2learn.service.model.response.RegisterResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/auth")
public class AuthenticationController {

    private final AuthenticationService authenticationService;

    @Autowired
    public AuthenticationController(AuthenticationService authenticationService) {
        this.authenticationService = authenticationService;
    }

    @PostMapping(path = "/register")
    public ResponseEntity<RegisterResponse> register(@RequestBody RegisterRequest request) {
        return new ResponseEntity<>(authenticationService.register(request), HttpStatus.OK);
    }

    @PostMapping
    public ResponseEntity<LoginResponse> authenticate(@RequestBody LoginRequest request) {
        return new ResponseEntity<>(authenticationService.login(request), HttpStatus.OK);
    }
}
