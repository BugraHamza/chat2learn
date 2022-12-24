package com.project.chat2learn.controller;

import com.project.chat2learn.service.ChatSessionService;
import com.project.chat2learn.service.model.dto.ChatSessionDTO;
import com.project.chat2learn.service.model.dto.ModelDTO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/chat")
public class ChatSessionController {

    private final ChatSessionService chatSessionService;

    @Autowired
    public ChatSessionController(ChatSessionService chatSessionService) {
        this.chatSessionService = chatSessionService;
    }

    @GetMapping
    public ResponseEntity<List<ChatSessionDTO>> getChatSession() {
        return new ResponseEntity<List<ChatSessionDTO>>(chatSessionService.getChatSessions(), HttpStatus.OK);
    }

    @PostMapping("/{modelId}")
    public ResponseEntity<ChatSessionDTO> createChatSession(@PathVariable Long modelId) {
        return new ResponseEntity<ChatSessionDTO>(chatSessionService.createChatSession(modelId), HttpStatus.CREATED);
    }

    @DeleteMapping(path = "/{id}")
    public ResponseEntity<ChatSessionDTO> deleteChatSession(@PathVariable Long id) {
        return new ResponseEntity<ChatSessionDTO>(chatSessionService.deleteChatSession(id), HttpStatus.OK);
    }
}
