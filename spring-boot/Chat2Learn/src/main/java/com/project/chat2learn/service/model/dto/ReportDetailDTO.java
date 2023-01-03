package com.project.chat2learn.service.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ReportDetailDTO {

    private Long messageCount;

    private Long errorCount;

    private Double averageScore;

    private List<ReportErrorCountDTO> reportErrorCountDTOList;

    private Map<LocalDate,Double> scoreMap;
}
