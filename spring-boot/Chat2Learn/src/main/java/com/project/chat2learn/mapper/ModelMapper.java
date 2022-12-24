package com.project.chat2learn.mapper;

import com.project.chat2learn.dao.domain.Model;
import com.project.chat2learn.service.model.dto.ModelDTO;
import org.mapstruct.*;

import java.util.List;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface ModelMapper {
    Model modelDTOToModel(ModelDTO modelDTO);

    ModelDTO modelToModelDTO(Model model);

    @BeanMapping(nullValuePropertyMappingStrategy = NullValuePropertyMappingStrategy.IGNORE)
    Model updateModelFromModelDTO(ModelDTO modelDTO, @MappingTarget Model model);

    List<ModelDTO> map2ModelDTOs(List<Model> models);
}
