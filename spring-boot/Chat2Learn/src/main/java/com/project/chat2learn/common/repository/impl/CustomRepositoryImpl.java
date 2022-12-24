package com.project.chat2learn.common.repository.impl;

import com.project.chat2learn.common.repository.CustomRepository;
import jdk.nashorn.internal.ir.CallNode;
import org.springframework.data.jpa.repository.support.JpaEntityInformation;
import org.springframework.data.jpa.repository.support.SimpleJpaRepository;
import org.springframework.data.repository.NoRepositoryBean;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.Assert;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;
import java.io.Serializable;

@NoRepositoryBean
public class CustomRepositoryImpl<T, ID extends Serializable> extends SimpleJpaRepository<T, ID>
        implements CustomRepository<T, ID> {

    private final EntityManager entityManager;

    public CustomRepositoryImpl(JpaEntityInformation entityInformation, EntityManager entityManager) {
        super(entityInformation, entityManager);
        this.entityManager = entityManager;
    }


    @Override
    @Transactional
    public void refresh(T t) {
        entityManager.refresh(t);
    }
}
