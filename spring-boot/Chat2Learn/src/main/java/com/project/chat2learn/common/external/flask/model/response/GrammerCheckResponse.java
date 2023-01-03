package com.project.chat2learn.common.external.flask.model.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class GrammerCheckResponse {

    private String correctText;

    private String taggedCorrectText;

    private List<ErrorType> errorTypes;

    private Double score;
}
