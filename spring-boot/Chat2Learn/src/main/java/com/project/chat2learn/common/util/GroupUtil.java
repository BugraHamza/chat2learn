package com.project.chat2learn.common.util;

import com.project.chat2learn.common.enums.IntervalType;
import com.project.chat2learn.dao.domain.GrammerError;
import com.project.chat2learn.dao.domain.Message;
import com.project.chat2learn.service.model.dto.GrammerErrorDTO;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.dto.ReportDetailDTO;
import jdk.nashorn.internal.ir.annotations.Immutable;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.temporal.TemporalAdjuster;
import java.time.temporal.TemporalAdjusters;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class GroupUtil {

    public static Map<LocalDate, List<MessageDTO>> groupMessages(IntervalType intervalType, List<MessageDTO> messages) {

        final Map<IntervalType, TemporalAdjuster> ADJUSTERS = new HashMap<>();

        ADJUSTERS.put(IntervalType.DAY, TemporalAdjusters.ofDateAdjuster(d -> d)); // identity
        ADJUSTERS.put(IntervalType.WEEK, TemporalAdjusters.previousOrSame(DayOfWeek.of(1)));
        ADJUSTERS.put(IntervalType.MONTH, TemporalAdjusters.firstDayOfMonth());
        ADJUSTERS.put(IntervalType.YEAR, TemporalAdjusters.firstDayOfYear());

        Map<LocalDate, List<MessageDTO>> result = messages.stream().collect(Collectors.groupingBy(item -> item.getCreatedDate().toLocalDate().with(ADJUSTERS.get(intervalType))));
        return result;
    }

    public static Map<LocalDate, ReportDetailDTO> map2ReportDetailDTO(Map<LocalDate, List<MessageDTO>> result) {
        Map<LocalDate, ReportDetailDTO> dateReportDetailDTOMap = new HashMap<>();
        result.forEach((localDate, messageList) -> {
            ReportDetailDTO reportDetailDTO = getReportDetailDTO(messageList);
            dateReportDetailDTOMap.put(localDate, reportDetailDTO);
        });
        return dateReportDetailDTOMap;
    }

    public static ReportDetailDTO getReportDetailDTO(List<MessageDTO> messageList) {
        Stream<MessageDTO> messageStream = messageList.stream();
        Long messageCount = messageStream.count();
        Stream<MessageDTO> errorDetectedMessages = messageStream.filter(message -> message.getReport() != null);
        Long errorCount = errorDetectedMessages.count();
        Map<GrammerErrorDTO, Long> grammerErrorDTOLongMap = errorDetectedMessages.flatMap(messages -> messages.getReport().getErrors().stream()).collect(Collectors.groupingBy(e -> e, Collectors.counting()));
        ReportDetailDTO reportDetailDTO = new ReportDetailDTO();
        reportDetailDTO.setMessageCount(messageCount);
        reportDetailDTO.setErrorCount(errorCount);
        reportDetailDTO.setGrammerErrorMap(grammerErrorDTOLongMap);
        return reportDetailDTO;
    }


}
