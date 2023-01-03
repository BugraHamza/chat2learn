package com.project.chat2learn.service;

import com.project.chat2learn.common.enums.IntervalType;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.dto.ReportDetailDTO;
import org.springframework.data.domain.Page;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

public interface ReportService {

    ReportDetailDTO getSessionReport(Long chatSessionId);

    ReportDetailDTO getAllSessionsReport();

    Page<MessageDTO> getMessagesByErrorType(String errorType, Integer page);



}
