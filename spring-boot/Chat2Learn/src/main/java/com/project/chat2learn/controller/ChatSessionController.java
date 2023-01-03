package com.project.chat2learn.controller;

import com.project.chat2learn.service.ChatSessionService;
import com.project.chat2learn.service.model.dto.ChatSessionDTO;
import com.project.chat2learn.service.model.dto.ModelDTO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/chat")
@Tag(name = "Chat Session Controller", description = "Endpoints for chat session")
public class ChatSessionController {

    private final ChatSessionService chatSessionService;

    @Autowired
    public ChatSessionController(ChatSessionService chatSessionService) {
        this.chatSessionService = chatSessionService;
    }

    @GetMapping
    @Operation(summary = "Get all chat sessions")
    public ResponseEntity<List<ChatSessionDTO>> getChatSession() {
        return new ResponseEntity<List<ChatSessionDTO>>(chatSessionService.getChatSessions(), HttpStatus.OK);
    }

    @PostMapping("/{modelId}")
    @Operation(summary = "Create a new chat session")
    public ResponseEntity<ChatSessionDTO> createChatSession(@PathVariable Long modelId) {
        return new ResponseEntity<ChatSessionDTO>(chatSessionService.createChatSession(modelId), HttpStatus.CREATED);
    }

    @DeleteMapping(path = "/{id}")
    @Operation(summary = "Delete a chat session")
    public ResponseEntity<ChatSessionDTO> deleteChatSession(@PathVariable Long id) {
        return new ResponseEntity<ChatSessionDTO>(chatSessionService.deleteChatSession(id), HttpStatus.OK);
    }
}
