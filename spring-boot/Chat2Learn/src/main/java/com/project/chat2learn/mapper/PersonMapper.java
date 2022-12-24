package com.project.chat2learn.mapper;

import com.project.chat2learn.dao.domain.Person;
import com.project.chat2learn.service.model.dto.PersonDTO;
import org.mapstruct.*;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface PersonMapper {
    Person personDTOToPerson(PersonDTO personDTO);

    PersonDTO personToPersonDTO(Person person);

    @BeanMapping(nullValuePropertyMappingStrategy = NullValuePropertyMappingStrategy.IGNORE)
    Person updatePersonFromPersonDTO(PersonDTO personDTO, @MappingTarget Person person);

    @AfterMapping
    default void linkSessions(@MappingTarget Person person) {
        person.getSessions().forEach(session -> session.setPerson(person));
    }
}
