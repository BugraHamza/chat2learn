package com.project.chat2learn.common.components;

import com.project.chat2learn.dao.domain.Person;
import com.project.chat2learn.security.model.UserDetailsImpl;
import org.springframework.data.domain.AuditorAware;
import org.springframework.security.authentication.AnonymousAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;

import java.util.Optional;

@Component
public class AuditAware implements AuditorAware <String> {

    @Override
    public Optional<String> getCurrentAuditor() {
        SecurityContext context = SecurityContextHolder.getContext();
        Authentication authentication = context.getAuthentication();
        if (authentication instanceof AnonymousAuthenticationToken) {
            return Optional.of("SYSTEM");
        }else {
            UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();
            return Optional.of(userDetails.getEmail());
        }
    }
}
