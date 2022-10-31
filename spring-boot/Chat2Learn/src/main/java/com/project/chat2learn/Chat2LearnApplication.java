package com.project.chat2learn;

import com.project.chat2learn.common.repository.impl.CustomRepositoryImpl;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@SpringBootApplication
@EnableJpaAuditing
@EnableFeignClients
@EnableJpaRepositories (repositoryBaseClass = CustomRepositoryImpl.class)
public class Chat2LearnApplication {

	public static void main(String[] args) {
		SpringApplication.run(Chat2LearnApplication.class, args);
	}



}
