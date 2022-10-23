package com.project.chat2learn.dao;

import lombok.Data;

import javax.persistence.*;
import java.util.Set;

@Entity
@Table(name = "report")
@Data
public class Report {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id", nullable = false)
    private Long id;

    @OneToOne(cascade = CascadeType.ALL)
    private Message message;

    private String correct_text;

    @ManyToMany
    @JoinTable(
            name = "report_error",
            joinColumns = @JoinColumn(name = "report_id"),
            inverseJoinColumns = @JoinColumn(name = "grammer_error_id"))
    Set<GrammerError> errors;

}