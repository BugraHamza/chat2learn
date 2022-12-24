package com.project.chat2learn.security.service;

import com.project.chat2learn.common.exception.ApiRequestException;
import com.project.chat2learn.dao.domain.Person;
import com.project.chat2learn.dao.repository.PersonRepository;
import com.project.chat2learn.security.model.UserDetailsImpl;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
public class UserDetailsServiceImpl implements UserDetailsService {

    private PersonRepository personRepository;

    public UserDetailsServiceImpl(PersonRepository personRepository) {
        super();
        this.personRepository = personRepository;
    }

    @Override
    public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
        Person person = personRepository.findByEmail(email)
                .orElseThrow(() -> new ApiRequestException("Person Not Found with email: " + email, HttpStatus.NOT_FOUND));

        return UserDetailsImpl.build(person);
    }
}
