package com.project.chat2learn.dao.repository;

import com.project.chat2learn.dao.domain.Message;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MessageRepository extends JpaRepository<Message, Long> {
}