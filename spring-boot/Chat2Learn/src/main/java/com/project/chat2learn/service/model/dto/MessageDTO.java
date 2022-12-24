package com.project.chat2learn.service.model.dto;

import com.project.chat2learn.common.enums.SenderType;
import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * A DTO for the {@link com.project.chat2learn.dao.domain.Message} entity
 */
@Data
public class MessageDTO implements Serializable {
    private final String createdBy;
    private final LocalDateTime createdDate;
    private final String lastModifiedBy;
    private final LocalDateTime lastModifiedDate;
    private final Long id;
    private final String text;
    private final SenderType senderType;
    private final ReportDTO report;
}