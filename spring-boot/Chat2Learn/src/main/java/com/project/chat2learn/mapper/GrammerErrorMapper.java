package com.project.chat2learn.mapper;

import com.project.chat2learn.dao.domain.ReportError;
import com.project.chat2learn.service.model.dto.ReportErrorDTO;
import org.mapstruct.*;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface GrammerErrorMapper {
    ReportError grammerErrorDTOToGrammerError(ReportErrorDTO reportErrorDTO);

    ReportErrorDTO grammerErrorToGrammerErrorDTO(ReportError reportError);

    @BeanMapping(nullValuePropertyMappingStrategy = NullValuePropertyMappingStrategy.IGNORE)
    ReportError updateGrammerErrorFromGrammerErrorDTO(ReportErrorDTO reportErrorDTO, @MappingTarget ReportError reportError);
}
