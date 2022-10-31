package com.project.chat2learn.service.model.dto;

import com.project.chat2learn.dao.domain.Message;
import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.Set;

/**
 * A DTO for the {@link com.project.chat2learn.dao.domain.ChatSession} entity
 */
@Data
public class ChatSessionDTO implements Serializable {
    private final String createdBy;
    private final LocalDateTime createdDate;
    private final String lastModifiedBy;
    private final LocalDateTime lastModifiedDate;
    private final Long id;
    private final Map<String, Object> state;
    private final PersonDTO person;
    private final ModelDTO model;
    private Set<MessageDTO> messages;
}