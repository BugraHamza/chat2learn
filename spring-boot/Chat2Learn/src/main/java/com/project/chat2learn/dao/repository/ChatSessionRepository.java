package com.project.chat2learn.dao.repository;

import com.project.chat2learn.common.repository.CustomRepository;
import com.project.chat2learn.dao.domain.ChatSession;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ChatSessionRepository extends JpaRepository<ChatSession, Long>, CustomRepository<ChatSession,Long> {

    List<ChatSession> findAllByPersonId(Long personId);
}