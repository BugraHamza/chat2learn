package com.project.chat2learn.service.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@EqualsAndHashCode(of = "code")
public class ReportErrorCountDTO {
    private String code;
    private String description;
    private Long count;
}
