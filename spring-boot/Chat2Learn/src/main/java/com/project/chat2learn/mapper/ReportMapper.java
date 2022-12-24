package com.project.chat2learn.mapper;

import com.project.chat2learn.dao.domain.Report;
import com.project.chat2learn.service.model.dto.ReportDTO;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface ReportMapper {

    ReportDTO map2ReportDTO(Report report);

    Report map2Report(ReportDTO reportDTO);
}
