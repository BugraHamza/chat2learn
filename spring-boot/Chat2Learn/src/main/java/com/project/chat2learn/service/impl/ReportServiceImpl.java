package com.project.chat2learn.service.impl;

import com.project.chat2learn.common.enums.IntervalType;
import com.project.chat2learn.common.util.GroupUtil;
import com.project.chat2learn.dao.domain.Message;
import com.project.chat2learn.dao.domain.Person;
import com.project.chat2learn.dao.repository.MessageRepository;
import com.project.chat2learn.mapper.MessageMapper;
import com.project.chat2learn.service.ReportService;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.dto.ReportDetailDTO;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Autowired;
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
    public Map<LocalDate, ReportDetailDTO> getSessionReport(Long chatSessionId, IntervalType intervalType) {
        log.info("Getting report for chat session with id: {}", chatSessionId);
        List<Message> messages = messageRepository.findAllByChatSessionId(chatSessionId);
        Map<LocalDate, List<MessageDTO>> groupedByLocalDate = GroupUtil.groupMessages( intervalType, mapper.mapToMessageDTOList(messages));
        Map<LocalDate, ReportDetailDTO> dateReportDetailDTOMap = GroupUtil.map2ReportDetailDTO(groupedByLocalDate);
        return dateReportDetailDTOMap;


    }


    @Override
    public ReportDetailDTO getAllSessionsReport() {
        log.info("Getting report for all chat sessions");
        Person person = (Person) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        List<Message> messages = messageRepository.findAllByChatSessionPersonId(person.getId());
        return GroupUtil.getReportDetailDTO(mapper.mapToMessageDTOList(messages));
    }
}
