package com.project.chat2learn.common.components;

import com.project.chat2learn.dao.domain.Person;
import org.springframework.data.domain.AuditorAware;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

import java.util.Optional;

@Component
public class AuditAware implements AuditorAware <Long> {

    @Override
    public Optional<Long> getCurrentAuditor() {
        Person principal = (Person) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        return Optional.of(principal.getId());
    }
}
