package com.project.chat2learn.controller;

import com.project.chat2learn.common.enums.IntervalType;
import com.project.chat2learn.service.ReportService;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.dto.ReportDetailDTO;
import io.swagger.v3.oas.annotations.Hidden;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/report")
@Tag(name = "Report Controller", description = "Endpoints for report")
public class ReportController {

    private final ReportService reportService;

    @Autowired
    public ReportController(ReportService reportService) {
        this.reportService = reportService;
    }

    @GetMapping
    @Operation(summary = "Get report for all chat session messages")
    public ResponseEntity<ReportDetailDTO> getReport() {
        return new ResponseEntity<>(reportService.getAllSessionsReport(), HttpStatus.OK);
    }

    @GetMapping(path = "/{chatSessionId}")
    @Operation(summary = "Get report for a chat session")
    public ResponseEntity<ReportDetailDTO> getReport(@PathVariable Long chatSessionId) {
        return new ResponseEntity<>(reportService.getSessionReport(chatSessionId), HttpStatus.OK);
    }

    @GetMapping( path = "/errors")
    @Operation(summary = "Get error messages by error type")
    public ResponseEntity<Page<MessageDTO>> getReport(@RequestParam String errorType, @RequestParam Integer page) {
        return new ResponseEntity<>(reportService.getMessagesByErrorType(errorType,page), HttpStatus.OK);
    }

}
