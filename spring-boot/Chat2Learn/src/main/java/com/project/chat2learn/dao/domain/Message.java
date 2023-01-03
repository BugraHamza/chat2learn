package com.project.chat2learn.dao.domain;

import com.project.chat2learn.common.enums.SenderType;
import com.project.chat2learn.common.model.Auditable;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.*;


@Entity
@Table(name = "message")
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class Message extends Auditable {
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "chat_session_id")
    private ChatSession chatSession;

    @Column(length = 1000)
    private String text;

    private Double score;

    @Enumerated(EnumType.STRING)
    private SenderType senderType;

    @OneToOne(cascade = CascadeType.ALL,fetch = FetchType.LAZY)
    private Report report;


}