package com.project.chat2learn.service.model.dto;

import com.project.chat2learn.dao.domain.ReportError;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * A DTO for the {@link ReportError} entity
 */
@Data
@EqualsAndHashCode(of = "code")
public class ReportErrorDTO implements Serializable {
    private final String createdBy;
    private final LocalDateTime createdDate;
    private final String lastModifiedBy;
    private final LocalDateTime lastModifiedDate;
    private final String code;
    private final String description;

}