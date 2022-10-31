package com.project.chat2learn.mapper;

import com.project.chat2learn.dao.domain.Message;
import com.project.chat2learn.service.model.dto.MessageDTO;
import org.mapstruct.*;

import java.util.List;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface MessageMapper {

    Message messageDTOToMessage(MessageDTO messageDTO);

    MessageDTO messageToMessageDTO(Message message);

    List<MessageDTO> mapToMessageDTOList(List<Message> messages);

    @BeanMapping(nullValuePropertyMappingStrategy = NullValuePropertyMappingStrategy.IGNORE)
    Message updateMessageFromMessageDTO(MessageDTO messageDTO, @MappingTarget Message message);
}
