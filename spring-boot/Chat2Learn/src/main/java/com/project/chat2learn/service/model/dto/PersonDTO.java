package com.project.chat2learn.service.model.dto;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Set;

/**
 * A DTO for the {@link com.project.chat2learn.dao.domain.Person} entity
 */
@Data
public class PersonDTO implements Serializable {
    private final String createdBy;
    private final LocalDateTime createdDate;
    private final String lastModifiedBy;
    private final LocalDateTime lastModifiedDate;
    private final Long id;
    private final String name;
    private final String lastname;
    private final String email;

}