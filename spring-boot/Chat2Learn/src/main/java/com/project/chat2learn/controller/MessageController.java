package com.project.chat2learn.controller;

import com.project.chat2learn.service.MessageService;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.request.CreateMessageRequest;
import com.project.chat2learn.service.model.response.CreateMessageResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/chat")
public class MessageController {

    private final MessageService messageService;

    @Autowired
    public MessageController(MessageService messageService) {
        this.messageService = messageService;
    }

    @GetMapping(path = "/{id}/message")
    public ResponseEntity<Page<MessageDTO>> getMessages(@PathVariable Long id, @RequestParam Integer page) {
        return new ResponseEntity<Page<MessageDTO>>(messageService.getMessages(id,page), HttpStatus.OK);
    }

    @PostMapping(path = "/{id}/message")
    public ResponseEntity<CreateMessageResponse> createMessage(@PathVariable Long id, @RequestBody CreateMessageRequest request) {
        return new ResponseEntity<CreateMessageResponse>(messageService.createMessage(id,request.getMessage()), HttpStatus.CREATED);
    }
}