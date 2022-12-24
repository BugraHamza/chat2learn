package com.project.chat2learn.service.model.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class RegisterRequest {

    private String name;

    private String lastname;

    private String email;

    private String password;

    private String confirmPassword;
}
