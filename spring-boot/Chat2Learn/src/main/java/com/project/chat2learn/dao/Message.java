package com.project.chat2learn.dao;

import lombok.Data;
import org.springframework.data.jpa.domain.AbstractAuditable;

import javax.persistence.*;

import com.project.chat2learn.common.enums.SenderType;


@Entity
@Table(name = "message")
@Data
public class Message extends AbstractAuditable {
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "chat_session_id")
    private ChatSession chatSession;

    private String text;

    @Enumerated(EnumType.STRING)
    private SenderType senderType;

    @OneToOne(cascade = CascadeType.ALL)
    private Report report;


}