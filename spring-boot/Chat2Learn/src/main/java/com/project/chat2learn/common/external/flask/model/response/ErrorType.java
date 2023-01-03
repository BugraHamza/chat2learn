package com.project.chat2learn.common.external.flask.model.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class ErrorType {

    private String code;

    private String description;
}
