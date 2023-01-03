package com.project.chat2learn.service.model.dto;

import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.List;

/**
 * A DTO for the {@link com.project.chat2learn.dao.domain.Report} entity
 */
@Data
public class ReportDTO implements Serializable {
    private final String createdBy;
    private final LocalDateTime createdDate;
    private final String lastModifiedBy;
    private final LocalDateTime lastModifiedDate;
    private final Long id;
    private final String correctText;
    private final String taggedCorrectText;
    private final List<ReportErrorDTO> errors;
}