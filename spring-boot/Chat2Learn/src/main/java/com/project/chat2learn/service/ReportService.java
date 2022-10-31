package com.project.chat2learn.service;

import com.project.chat2learn.common.enums.IntervalType;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.dto.ReportDetailDTO;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

public interface ReportService {

    Map<LocalDate, ReportDetailDTO> getSessionReport(Long chatSessionId, IntervalType intervalType);

    ReportDetailDTO getAllSessionsReport();



}
