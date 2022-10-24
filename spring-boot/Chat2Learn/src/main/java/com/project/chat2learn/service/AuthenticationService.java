package com.project.chat2learn.service;

import com.project.chat2learn.service.model.request.LoginRequest;
import com.project.chat2learn.service.model.request.RegisterRequest;
import com.project.chat2learn.service.model.response.LoginResponse;
import com.project.chat2learn.service.model.response.RegisterResponse;

public interface AuthenticationService {

    RegisterResponse register(RegisterRequest request);

    LoginResponse login(LoginRequest request);
}
