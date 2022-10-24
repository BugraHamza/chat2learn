package com.project.chat2learn.dao.repository;

import com.project.chat2learn.dao.domain.Report;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ReportRepository extends JpaRepository<Report, Long> {
}