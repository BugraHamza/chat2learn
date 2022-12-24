package com.project.chat2learn.dao.repository;

import com.project.chat2learn.dao.domain.GrammerError;
import org.springframework.data.jpa.repository.JpaRepository;

public interface GrammerErrorRepository extends JpaRepository<GrammerError, Long> {
}