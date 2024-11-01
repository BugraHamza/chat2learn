package com.project.chat2learn.controller;

import com.project.chat2learn.service.ModelService;
import com.project.chat2learn.service.model.dto.ModelDTO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/model")
@Tag(name = "Model Controller", description = "Endpoints for model")
public class ModelController {

    private final ModelService modelService;

    @Autowired
    public ModelController(ModelService modelService) {
        this.modelService = modelService;
    }

    @GetMapping
    @Operation(summary = "Get all models")
    public ResponseEntity<List<ModelDTO>> getAllModels() {
        return new ResponseEntity<List<ModelDTO>>(modelService.getAllModels(), HttpStatus.OK);
    }

    @PostMapping
    @Operation(summary = "Create a new model")
    public ResponseEntity<ModelDTO> createModel(@RequestBody ModelDTO modelDTO) {
        return new ResponseEntity<ModelDTO>(modelService.create(modelDTO), HttpStatus.CREATED);
    }
}
