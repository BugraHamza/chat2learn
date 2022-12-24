package com.project.chat2learn.dao.repository;

import com.project.chat2learn.dao.domain.Model;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ModelRepository extends JpaRepository<Model, Long> {
}