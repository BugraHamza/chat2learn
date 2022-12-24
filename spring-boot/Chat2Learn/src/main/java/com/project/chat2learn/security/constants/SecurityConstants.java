package com.project.chat2learn.security.constants;

public class SecurityConstants {

    public static final String SIGN_UP_URL = "/api/v1/user/register";
    public static final String SECRET = "chat2learn";
    public static final long EXPIRATION_TIME = 432_000_000; // 5 gün
    public static final String TOKEN_PREFIX = "Bearer ";
    public static final String HEADER_STRING = "Authorization";
}
