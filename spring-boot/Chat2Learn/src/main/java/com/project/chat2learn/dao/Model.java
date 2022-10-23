package com.project.chat2learn.dao;

import lombok.Data;
import org.springframework.data.jpa.domain.AbstractAuditable;

import javax.persistence.*;

@Entity
@Data
@Table(name = "model")
public class Model extends AbstractAuditable {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    private Long id;

    private String name;

    @OneToOne(cascade = CascadeType.ALL)
    ChatSession chatSession;

}