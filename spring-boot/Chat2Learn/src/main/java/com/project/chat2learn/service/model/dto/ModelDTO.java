package com.project.chat2learn.service.model.dto;

import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Set;

/**
 * A DTO for the {@link com.project.chat2learn.dao.domain.Model} entity
 */
@Data
public class ModelDTO implements Serializable {
    private final String createdBy;
    private final LocalDateTime createdDate;
    private final String lastModifiedBy;
    private final LocalDateTime lastModifiedDate;
    private final Long id;
    private final String name;
}