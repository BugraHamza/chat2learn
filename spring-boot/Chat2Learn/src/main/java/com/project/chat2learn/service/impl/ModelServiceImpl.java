package com.project.chat2learn.service.impl;

import com.project.chat2learn.dao.repository.ModelRepository;
import com.project.chat2learn.mapper.IModelMapper;
import com.project.chat2learn.service.ModelService;
import com.project.chat2learn.service.model.dto.ModelDTO;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@Log4j2
public class ModelServiceImpl implements ModelService {

    private final ModelRepository modelRepository;
    private final IModelMapper mapper;

    @Autowired
    public ModelServiceImpl(ModelRepository modelRepository, IModelMapper mapper) {
        this.modelRepository = modelRepository;
        this.mapper = mapper;
    }

    @Override
    public List<ModelDTO> getAllModels() {
        return mapper.map2ModelDTOs(modelRepository.findAll());
    }

    @Override
    public ModelDTO create(ModelDTO modelDTO) {

        return mapper.modelToModelDTO(modelRepository.save(mapper.modelDTOToModel(modelDTO)));
    }
}
