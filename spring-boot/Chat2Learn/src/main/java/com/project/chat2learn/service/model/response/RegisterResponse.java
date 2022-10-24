package com.project.chat2learn.service.model.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class RegisterResponse {

    private Long id;

    private String name;

    private String lastname;

    private String email;

    private String token;
}
