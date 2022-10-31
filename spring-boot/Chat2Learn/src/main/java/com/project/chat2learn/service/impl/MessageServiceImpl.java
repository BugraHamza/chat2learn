package com.project.chat2learn.service.impl;

import com.project.chat2learn.common.enums.SenderType;
import com.project.chat2learn.common.exception.ApiRequestException;
import com.project.chat2learn.common.external.flask.model.response.BaseResponse;
import com.project.chat2learn.common.external.flask.service.BotService;
import com.project.chat2learn.dao.domain.ChatSession;
import com.project.chat2learn.dao.domain.GrammerError;
import com.project.chat2learn.dao.domain.Message;
import com.project.chat2learn.dao.domain.Report;
import com.project.chat2learn.dao.repository.ChatSessionRepository;
import com.project.chat2learn.dao.repository.GrammerErrorRepository;
import com.project.chat2learn.dao.repository.MessageRepository;
import com.project.chat2learn.mapper.MessageMapper;
import com.project.chat2learn.service.GrammerErrorService;
import com.project.chat2learn.service.MessageService;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.response.CreateMessageResponse;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@Log4j2
public class MessageServiceImpl implements MessageService {
    private  final MessageRepository messageRepository;
    private final ChatSessionRepository chatSessionRepository;
    private final GrammerErrorRepository grammerErrorRepository;
    private final BotService botService;
    private final MessageMapper mapper;

    @Autowired
    public MessageServiceImpl(MessageRepository messageRepository, MessageMapper mapper, ChatSessionRepository chatSessionRepository, BotService botService,GrammerErrorRepository grammerErrorRepository) {
        this.messageRepository = messageRepository;
        this.mapper = mapper;
        this.chatSessionRepository = chatSessionRepository;
        this.botService = botService;
        this.grammerErrorRepository = grammerErrorRepository;
    }

    @Override
    public Page<MessageDTO> getMessages(Long chatSessionId, Integer page) {
        Pageable pageable = PageRequest.of(page, 10, Sort.by("createdDate").descending());
        Page<Message> messagePage = messageRepository.findAllByChatSessionId(chatSessionId, pageable);
        Page<MessageDTO> messageDTOPage = messagePage.map(mapper::messageToMessageDTO);
        return messageDTOPage;
    }

    @Override
    public CreateMessageResponse createMessage(Long chatSessionId, String message) {
        ChatSession session = chatSessionRepository.findById(chatSessionId).orElseThrow(() -> {
            log.error("Chat session with id {} not found", chatSessionId);
            return new ApiRequestException("Chat session not found", HttpStatus.NOT_FOUND);
        });

        if(!StringUtils.hasLength(message)) {
            log.error("Message is empty");
            throw new ApiRequestException("Message is empty", HttpStatus.BAD_REQUEST);
        }

        BaseResponse response = botService.messageBotWithId(session.getModel().getId(), message);

        Message savedPersonMessage = createPersonMessage(message, session, response);

        Message savedBotMessage = createBotMessage(session, response);

        return createMessageResponse(savedPersonMessage, savedBotMessage);
    }

    private Message createBotMessage(ChatSession session, BaseResponse response) {
        Message botMessage = new Message();
        botMessage.setText(response.getResponseMessage());
        botMessage.setChatSession(session);
        botMessage.setSenderType(SenderType.BOT);

        Message savedBotMessage = messageRepository.save(botMessage);
        return savedBotMessage;
    }

    private Message createPersonMessage(String message, ChatSession session, BaseResponse response) {
        Message newMessage = new Message();
        newMessage.setText(message);
        newMessage.setChatSession(session);
        newMessage.setSenderType(SenderType.PERSON);

        Report report =  new Report();
        report.setMessage(newMessage);
        report.setCorrectText(response.getCorrectText());
        report.setErrors(grammerErrorListByIdList(response.getGrammarErrors()));

        newMessage.setReport(report);
        Message savedMessage = messageRepository.save(newMessage);
        return savedMessage;
    }

    private CreateMessageResponse createMessageResponse(Message savedPersonMessage, Message savedBotMessage) {
        CreateMessageResponse createMessageResponse = new CreateMessageResponse();
        createMessageResponse.setPersonMessage(mapper.messageToMessageDTO(savedPersonMessage));
        createMessageResponse.setBotMessage(mapper.messageToMessageDTO(savedBotMessage));
        return createMessageResponse;
    }
    private List<GrammerError> grammerErrorListByIdList(Set<Long> idList){
        if(CollectionUtils.isEmpty(idList)){
            log.warn("Id list is empty");
            return new ArrayList<>();
        }
        return grammerErrorRepository.findAllById(idList);
    }

}
