package com.project.chat2learn.dao;

import com.project.chat2learn.common.converter.HashMapConverter;
import lombok.Data;
import org.springframework.data.jpa.domain.AbstractAuditable;

import javax.persistence.*;
import java.util.Map;
import java.util.Set;

@Entity
@Table(name = "chat_session")
@Data
public class ChatSession extends AbstractAuditable {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    Long id;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;

    @OneToOne(cascade = CascadeType.ALL)
    private Model model;

    @Convert(converter = HashMapConverter.class)
    private Map<String, Object> state;

    @OneToMany(mappedBy="chat_session", cascade = CascadeType.ALL)
    private Set<Message> messages;






}