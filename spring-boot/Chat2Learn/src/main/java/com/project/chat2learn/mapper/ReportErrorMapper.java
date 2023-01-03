package com.project.chat2learn.mapper;

import com.project.chat2learn.dao.domain.Report;
import com.project.chat2learn.service.model.dto.ReportDTO;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface ReportErrorMapper {

    Report reportDTOToReport(ReportDTO reportDTO);

    ReportDTO reportToReportDTO(Report report);
}
