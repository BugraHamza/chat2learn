package com.project.chat2learn.mapper;

import com.project.chat2learn.dao.domain.ChatSession;
import com.project.chat2learn.service.model.dto.ChatSessionDTO;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

import java.util.List;
import java.util.Set;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface ChatSessionMapper {

    ChatSessionDTO map2ChatSessionDTO(ChatSession chatSession);

    ChatSession map2ChatSession(ChatSessionDTO chatSessionDTO);

    List<ChatSessionDTO> map2ChatSessionDTOs(List<ChatSession> chatSessions);
}
