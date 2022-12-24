package com.project.chat2learn.common.external.flask.model.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Set;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class BaseResponse {

    private String correctText;

    private String responseMessage;

    private Set<Long> grammarErrors;
}
