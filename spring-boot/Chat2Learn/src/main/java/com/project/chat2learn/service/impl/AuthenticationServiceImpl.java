package com.project.chat2learn.service.impl;

import com.project.chat2learn.common.exception.ApiRequestException;
import com.project.chat2learn.dao.domain.Person;
import com.project.chat2learn.dao.repository.PersonRepository;
import com.project.chat2learn.security.jwt.JwtUtils;
import com.project.chat2learn.security.model.UserDetailsImpl;
import com.project.chat2learn.service.AuthenticationService;
import com.project.chat2learn.service.model.request.LoginRequest;
import com.project.chat2learn.service.model.request.RegisterRequest;
import com.project.chat2learn.service.model.response.LoginResponse;
import com.project.chat2learn.service.model.response.RegisterResponse;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.validator.routines.EmailValidator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@Log4j2
public class AuthenticationServiceImpl implements AuthenticationService {

    private final PersonRepository personRepository;
    private final AuthenticationManager authenticationManager;
    private final PasswordEncoder encoder;
    private final JwtUtils jwtUtils;

    @Autowired
    public AuthenticationServiceImpl(PersonRepository personRepository, AuthenticationManager authenticationManager, PasswordEncoder encoder, JwtUtils jwtUtils) {
        this.personRepository = personRepository;
        this.authenticationManager = authenticationManager;
        this.encoder = encoder;
        this.jwtUtils = jwtUtils;
    }

    @Override
    public RegisterResponse register(RegisterRequest request) {
        validateEmailAndPassword(request.getEmail(), request.getPassword(), request.getConfirmPassword());

        Person person = personRepository.findByEmail(request.getEmail()).orElse(null);

        if(person == null ) {
            throw new ApiRequestException("Please make sure you got the OTP before registering!",HttpStatus.BAD_REQUEST);

        }


        person.setPassword(encoder.encode(request.getPassword()));
        person.setName(request.getName());
        person.setLastname(request.getLastname());
        person.setEmail(request.getEmail());
        Person registeredPerson = personRepository.save(person);

        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(registeredPerson.getEmail(), registeredPerson.getPassword()));

        SecurityContextHolder.getContext().setAuthentication(authentication);
        String jwt = jwtUtils.generateJwtToken(authentication);

        UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();
        return new RegisterResponse (userDetails.getId(), userDetails.getName(), userDetails.getLastname(),userDetails.getEmail(),jwt);
    }

    @Override
    public LoginResponse login(LoginRequest request) {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(request.getEmail(), request.getPassword()));

        SecurityContextHolder.getContext().setAuthentication(authentication);
        String jwt = jwtUtils.generateJwtToken(authentication);

        UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();

        return new LoginResponse(userDetails.getId(), userDetails.getName(), userDetails.getLastname(),userDetails.getEmail(),jwt);
    }
    private void validateEmailAndPassword(String email ,String password, String confirmPassword) {

        EmailValidator emailValidator = EmailValidator.getInstance();

        if(!emailValidator.isValid(email)) {
            throw new ApiRequestException("Given email is not valid.", HttpStatus.BAD_REQUEST);

        }
        if(password == null || password.length() < 8) {
            throw new ApiRequestException("Given password is not secure. Password length must be at least 6!", HttpStatus.BAD_REQUEST);
        }

        if(!password.equals(confirmPassword)) {
            throw new ApiRequestException("Given passwords not matching !", HttpStatus.BAD_REQUEST);
        }

    }
}
