package com.project.chat2learn;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;

@SpringBootApplication
@EnableJpaAuditing
@EnableFeignClients
public class Chat2LearnApplication {

	public static void main(String[] args) {
		SpringApplication.run(Chat2LearnApplication.class, args);
	}



}
