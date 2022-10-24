package com.project.chat2learn.dao.domain;

import com.project.chat2learn.common.model.Auditable;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.*;
import java.util.Set;

@Entity
@Table(name = "person")
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class Person extends Auditable {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    private Long id;

    private String name;

    private String lastname;

    private String email;

    private String password;

    @OneToMany(mappedBy="person",cascade = CascadeType.ALL,fetch = FetchType.LAZY)
    private Set<ChatSession> sessions;



}