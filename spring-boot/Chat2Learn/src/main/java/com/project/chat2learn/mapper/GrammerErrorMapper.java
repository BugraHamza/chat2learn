package com.project.chat2learn.mapper;

import com.project.chat2learn.dao.domain.GrammerError;
import com.project.chat2learn.service.model.dto.GrammerErrorDTO;
import org.mapstruct.*;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface GrammerErrorMapper {
    GrammerError grammerErrorDTOToGrammerError(GrammerErrorDTO grammerErrorDTO);

    GrammerErrorDTO grammerErrorToGrammerErrorDTO(GrammerError grammerError);

    @BeanMapping(nullValuePropertyMappingStrategy = NullValuePropertyMappingStrategy.IGNORE)
    GrammerError updateGrammerErrorFromGrammerErrorDTO(GrammerErrorDTO grammerErrorDTO, @MappingTarget GrammerError grammerError);
}
