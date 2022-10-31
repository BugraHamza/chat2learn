package com.project.chat2learn.controller;

import com.project.chat2learn.common.enums.IntervalType;
import com.project.chat2learn.service.ReportService;
import com.project.chat2learn.service.model.dto.ReportDetailDTO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.Map;

@RestController
@RequestMapping("/report")
public class ReportController {

    private final ReportService reportService;

    @Autowired
    public ReportController(ReportService reportService) {
        this.reportService = reportService;
    }

    @GetMapping
    public ResponseEntity<ReportDetailDTO> getReport() {
        return new ResponseEntity<>(reportService.getAllSessionsReport(), HttpStatus.OK);
    }

    @GetMapping
    @RequestMapping("/{chatSessionId}")
    public ResponseEntity<Map<LocalDate, ReportDetailDTO>> getReport(@PathVariable Long chatSessionId, @RequestParam IntervalType intervalType) {
        return new ResponseEntity<>(reportService.getSessionReport(chatSessionId,intervalType), HttpStatus.OK);
    }

}
