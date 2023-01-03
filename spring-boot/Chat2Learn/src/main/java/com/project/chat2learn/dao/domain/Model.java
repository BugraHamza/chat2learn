package com.project.chat2learn.dao.domain;

import com.project.chat2learn.common.model.Auditable;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.*;
import java.util.HashSet;
import java.util.Set;

@Entity
@Table(name = "model")
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class Model {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    private Long id;

    private String name;

    private String description;

    @OneToMany(mappedBy="model",cascade = CascadeType.ALL,fetch = FetchType.LAZY)
    private Set<ChatSession> sessions =  new HashSet<>();

}