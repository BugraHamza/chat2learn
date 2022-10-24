package com.project.chat2learn.dao.domain;

import com.project.chat2learn.common.converter.HashMapConverter;
import com.project.chat2learn.common.model.Auditable;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.*;
import java.util.Map;
import java.util.Set;

@Entity
@Table(name = "chat_session")
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class ChatSession extends Auditable {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    Long id;

    @ManyToOne
    @JoinColumn(name = "person_id",nullable = false)
    private Person person;

    @OneToOne(cascade = CascadeType.ALL,fetch = FetchType.LAZY)
    private Model model;

    @Convert(converter = HashMapConverter.class)
    private Map<String, Object> state;

    @OneToMany(mappedBy="chatSession", cascade = CascadeType.ALL,fetch = FetchType.LAZY)
    private Set<Message> messages;






}