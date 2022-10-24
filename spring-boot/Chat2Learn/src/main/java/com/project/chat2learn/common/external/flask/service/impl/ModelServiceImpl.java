package com.project.chat2learn.common.external.flask.service.impl;

import com.project.chat2learn.common.external.flask.client.FlaskFeignClient;
import com.project.chat2learn.common.external.flask.service.ModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ModelServiceImpl implements ModelService {

    private final FlaskFeignClient client;

    @Autowired
    public ModelServiceImpl(FlaskFeignClient client) {
        this.client = client;
    }

    @Override
    public String getString(Long id) {
        String a =client.getString(id);
        return a;
    }
}
