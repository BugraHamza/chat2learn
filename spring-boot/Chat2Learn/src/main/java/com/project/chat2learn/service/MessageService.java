package com.project.chat2learn.service;

import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.response.CreateMessageResponse;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import java.util.Set;
import java.util.concurrent.ExecutionException;

public interface MessageService {

    Page<MessageDTO> getMessages(Long chatSessionId, Integer page);

    CreateMessageResponse createMessage(Long chatSessionId, String message);



}
