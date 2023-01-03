package com.project.chat2learn.common.exception;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

@ControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

    @ExceptionHandler(value = ApiRequestException.class)
    public ResponseEntity<ApiErrorResponse> DataNotFoundResponse(ApiRequestException apiRequestException) {
        return new ResponseEntity<ApiErrorResponse>(
                new ApiErrorResponse(
                        apiRequestException.getMessage(),
                        apiRequestException.getHttpStatus(),
                        apiRequestException.getTimestamp()
                ),
                apiRequestException.getHttpStatus());
    }
}
