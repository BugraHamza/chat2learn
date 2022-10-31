package com.project.chat2learn.service;

import com.project.chat2learn.service.model.dto.ModelDTO;

import java.util.List;

public interface ModelService {

    List<ModelDTO> getAllModels();

    ModelDTO create(ModelDTO modelDTO);
}
