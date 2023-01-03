package com.project.chat2learn;

import com.project.chat2learn.common.repository.impl.CustomRepositoryImpl;
import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;

@SpringBootApplication
@EnableJpaAuditing
@EnableAsync(proxyTargetClass = true)
@EnableFeignClients
@EnableJpaRepositories (repositoryBaseClass = CustomRepositoryImpl.class)
@OpenAPIDefinition(info = @Info(title = "Chat2Learn API", version = "1.0", description = "Chat2Learn API"))
public class Chat2LearnApplication {

	public static void main(String[] args) {
		SpringApplication.run(Chat2LearnApplication.class, args);
	}



}
