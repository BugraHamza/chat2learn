package com.project.chat2learn.dao;

import javax.persistence.*;
import java.util.Set;

@Entity
@Table(name = "grammer_error")
public class GrammerError {
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    private Long id;

    private String name;

    private String description;

    @ManyToMany(mappedBy = "errors")
    private Set<Report> reports;

}