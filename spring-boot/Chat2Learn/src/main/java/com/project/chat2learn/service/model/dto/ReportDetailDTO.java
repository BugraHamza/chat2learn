package com.project.chat2learn.service.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;
import java.util.Set;
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ReportDetailDTO {

    private Long messageCount;

    private Long errorCount;

    private Map<GrammerErrorDTO,Long> grammerErrorMap;
}
