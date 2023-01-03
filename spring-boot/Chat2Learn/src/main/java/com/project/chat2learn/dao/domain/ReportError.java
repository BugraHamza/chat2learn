package com.project.chat2learn.dao.domain;

import com.project.chat2learn.common.model.Auditable;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.*;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "grammer_error")
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class ReportError extends Auditable {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;

    private String code;

    private String description;
    @ManyToOne
    @JoinColumn(name = "report_id")
    private Report report;

}