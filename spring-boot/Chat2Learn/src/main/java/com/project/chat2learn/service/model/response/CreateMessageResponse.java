package com.project.chat2learn.service.model.response;

import com.project.chat2learn.service.model.dto.MessageDTO;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CreateMessageResponse {

    private MessageDTO personMessage;

    private MessageDTO botMessage;
}
