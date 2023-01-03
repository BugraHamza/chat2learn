package com.project.chat2learn.service.impl;

import com.project.chat2learn.common.enums.IntervalType;
import com.project.chat2learn.common.enums.SenderType;
import com.project.chat2learn.common.util.GroupUtil;
import com.project.chat2learn.dao.domain.Message;
import com.project.chat2learn.dao.domain.Person;
import com.project.chat2learn.dao.domain.Report;
import com.project.chat2learn.dao.repository.MessageRepository;
import com.project.chat2learn.mapper.MessageMapper;
import com.project.chat2learn.security.model.UserDetailsImpl;
import com.project.chat2learn.service.ReportService;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.dto.ReportDetailDTO;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@Service
@Log4j2
public class ReportServiceImpl implements ReportService {

    private final MessageRepository messageRepository;

    private final MessageMapper mapper;

    @Autowired
    public ReportServiceImpl(MessageRepository messageRepository,MessageMapper mapper) {
        this.messageRepository = messageRepository;
        this.mapper = mapper;
    }


    @Override
    public ReportDetailDTO getSessionReport(Long chatSessionId) {
        log.info("Getting report for chat session with id: {}", chatSessionId);
        List<Message> messages = messageRepository.findAllByChatSessionIdAndSenderType(chatSessionId,SenderType.PERSON);
        return GroupUtil.getReportDetailDTO(mapper.mapToMessageDTOList(messages));

    }


    @Override
    public ReportDetailDTO getAllSessionsReport() {
        log.info("Getting report for all chat sessions");
        UserDetailsImpl userDetails = (UserDetailsImpl) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        List<Message> messages = messageRepository.findAllByChatSessionPersonIdAndSenderType(userDetails.getId(), SenderType.PERSON);
        List<MessageDTO> messageDTOList = mapper.mapToMessageDTOList(messages);
        ReportDetailDTO reportDetailDTO=  GroupUtil.getReportDetailDTO(messageDTOList);
        reportDetailDTO.setScoreMap(GroupUtil.getScoreMap(messageDTOList));
        return reportDetailDTO;
    }

    @Override
    public Page<MessageDTO> getMessagesByErrorType(String errorType,Integer page) {
        log.info("Getting messages by error type: {}", errorType);
        UserDetailsImpl userDetails = (UserDetailsImpl) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        Pageable pageable = PageRequest.of(page, 10, Sort.by("createdDate").descending());
        Page<Message> messagePage = messageRepository.findAllByChatSessionPersonIdAndSenderTypeAndReportErrorsCode(userDetails.getId(),SenderType.PERSON,errorType,pageable);
        return messagePage.map(mapper::messageToMessageDTO);
    }
}
