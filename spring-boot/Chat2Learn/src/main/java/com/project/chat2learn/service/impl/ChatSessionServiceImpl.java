package com.project.chat2learn.service.impl;

import com.project.chat2learn.common.exception.ApiRequestException;
import com.project.chat2learn.dao.domain.ChatSession;
import com.project.chat2learn.dao.domain.Model;
import com.project.chat2learn.dao.domain.Person;
import com.project.chat2learn.dao.repository.ChatSessionRepository;
import com.project.chat2learn.mapper.ChatSessionMapper;
import com.project.chat2learn.security.model.UserDetailsImpl;
import com.project.chat2learn.service.ChatSessionService;
import com.project.chat2learn.service.model.dto.ChatSessionDTO;
import lombok.extern.log4j.Log4j2;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;
import javax.security.auth.login.Configuration;
import java.util.List;
import java.util.Set;

@Service
@Log4j2
public class ChatSessionServiceImpl implements ChatSessionService {

    private final ChatSessionRepository chatSessionRepository;

    private final ChatSessionMapper mapper;

    @Autowired
    public ChatSessionServiceImpl(ChatSessionRepository chatSessionRepository,ChatSessionMapper mapper) {
        this.chatSessionRepository = chatSessionRepository;
        this.mapper = mapper;
    }

    @Override
    public List<ChatSessionDTO> getChatSessions() {
        UserDetailsImpl userDetails = (UserDetailsImpl) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        return mapper.map2ChatSessionDTOs(chatSessionRepository.findAllByPersonId(userDetails.getId()));
    }

    @Override
    public ChatSessionDTO getChatSession(Long id) {

        ChatSession chatSession = chatSessionRepository.findById(id).orElseThrow(() -> {
            log.error("Chat session with id {} not found", id);
            return new ApiRequestException("Chat session not found", HttpStatus.NOT_FOUND);
        });

        return mapper.map2ChatSessionDTO(chatSession);
    }

    @Override
    public ChatSessionDTO createChatSession(Long modelId) {
        UserDetailsImpl userDetails = (UserDetailsImpl) SecurityContextHolder.getContext().getAuthentication().getPrincipal();

        Model model = new Model();
        model.setId(modelId);

        Person person =  new Person();
        person.setId(userDetails.getId());

        ChatSession chatSession = new ChatSession();
        chatSession.setPerson(person);
        chatSession.setModel(model);

        ChatSession createdChatSession = chatSessionRepository.save(chatSession);
        chatSessionRepository.refresh(createdChatSession);

        return mapper.map2ChatSessionDTO(createdChatSession);
    }

    @Override
    public ChatSessionDTO deleteChatSession(Long id) {
        ChatSession chatSession = chatSessionRepository.findById(id).orElseThrow(() -> {
            log.error("Chat session with id {} not found", id);
            return new ApiRequestException("Chat session not found", HttpStatus.NOT_FOUND);
        });

        chatSessionRepository.delete(chatSession);
        return mapper.map2ChatSessionDTO(chatSession);
    }
}
