package com.project.chat2learn.common.exception;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

@ControllerAdvice
public class GlobalExeptionHandler extends ResponseEntityExceptionHandler {

    @ExceptionHandler(value = ApiRequestException.class)
    public ResponseEntity<ApiRequestException> DataNotFoundResponse(ApiRequestException apiRequestException) {
        return new ResponseEntity<>(apiRequestException,apiRequestException.getHttpStatus());
    }
}
