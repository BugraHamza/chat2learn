package com.project.chat2learn.dao;

import lombok.Data;
import org.springframework.data.jpa.domain.AbstractAuditable;

import javax.persistence.*;
import java.util.Set;

@Data
@Entity
@Table(name = "user")
public class User extends AbstractAuditable {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    private Long id;

    private String name;

    private String lastname;

    private String email;

    private String password;

    @OneToMany(mappedBy="user", cascade = CascadeType.ALL)
    private Set<ChatSession> sessions;



}