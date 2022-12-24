package com.project.chat2learn.service;

import com.project.chat2learn.service.model.dto.ChatSessionDTO;

import java.util.List;

public interface ChatSessionService {

    List<ChatSessionDTO> getChatSessions();

    ChatSessionDTO getChatSession(Long id);

    ChatSessionDTO createChatSession(Long modelId);

    ChatSessionDTO deleteChatSession(Long id);

}
