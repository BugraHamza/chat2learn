package com.project.chat2learn.dao.repository;

import com.project.chat2learn.dao.domain.Message;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;

import java.util.List;

public interface MessageRepository extends JpaRepository<Message, Long>{

    Page<Message> findAllByChatSessionId(Long id,Pageable pageable);

    List<Message> findAllByChatSessionId(Long id);

    List<Message> findAllByChatSessionPersonId(Long id);

}