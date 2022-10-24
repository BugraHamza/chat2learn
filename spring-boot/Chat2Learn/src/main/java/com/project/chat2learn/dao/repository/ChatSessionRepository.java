package com.project.chat2learn.dao.repository;

import com.project.chat2learn.dao.domain.ChatSession;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ChatSessionRepository extends JpaRepository<ChatSession, Long> {
}