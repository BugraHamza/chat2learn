package com.project.chat2learn.common.util;

import com.project.chat2learn.common.enums.IntervalType;
import com.project.chat2learn.mapper.ReportErrorMapper;
import com.project.chat2learn.service.model.dto.MessageDTO;
import com.project.chat2learn.service.model.dto.ReportDetailDTO;
import com.project.chat2learn.service.model.dto.ReportErrorCountDTO;
import com.project.chat2learn.service.model.dto.ReportErrorDTO;
import org.mapstruct.factory.Mappers;

import java.text.DecimalFormat;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.temporal.TemporalAdjuster;
import java.time.temporal.TemporalAdjusters;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class GroupUtil {

    private static final DecimalFormat df = new DecimalFormat("0.00");

    private static ReportErrorMapper reportErrorMapper = Mappers.getMapper(ReportErrorMapper.class);
    public static Map<LocalDate, List<MessageDTO>> groupMessages(IntervalType intervalType, List<MessageDTO> messages) {

        final Map<IntervalType, TemporalAdjuster> ADJUSTERS = new HashMap<>();

        ADJUSTERS.put(IntervalType.DAY, TemporalAdjusters.ofDateAdjuster(d -> d)); // identity
        ADJUSTERS.put(IntervalType.WEEK, TemporalAdjusters.previousOrSame(DayOfWeek.of(1)));
        ADJUSTERS.put(IntervalType.MONTH, TemporalAdjusters.firstDayOfMonth());
        ADJUSTERS.put(IntervalType.YEAR, TemporalAdjusters.firstDayOfYear());


        return messages.stream()
                .collect(Collectors.groupingBy(m -> m.getCreatedDate().toLocalDate().with(ADJUSTERS.get(intervalType))));

    }

    public static Map<LocalDate, Double> getScoreMap(List<MessageDTO>messages){
        Map<LocalDate, List<MessageDTO>> groupedMessages = groupMessages(IntervalType.WEEK, messages);
        Map<LocalDate, Double> scoreMap = new HashMap<>();
        groupedMessages.forEach((k,v) -> {
            double score = v.stream().mapToDouble(MessageDTO::getScore).average().orElse(0.0);
            scoreMap.put(k, Double.parseDouble(df.format(score)));
        });
        return scoreMap;
    }


    public static List<ReportErrorCountDTO> getReportErrorCount(List<MessageDTO> messageList){
        List<MessageDTO> errorDetectedMessages = messageList.stream().filter(message -> message.getReport() != null).collect(Collectors.toList());
        Map<String, Long> reportErrorDTOLongMap = errorDetectedMessages.stream().flatMap(messages -> messages.getReport().getErrors().stream()).map(e -> e.getCode()).collect(Collectors.groupingBy(e -> e, Collectors.counting()));
        // Set<ReportErrorCountDTO> errorDetectedMessages.stream().map

        return null;
    }

    public static ReportDetailDTO getReportDetailDTO(List<MessageDTO> messageList) {
        Long messageCount = Long.valueOf(messageList.size());
        Double averageScore = messageList.stream().mapToDouble(MessageDTO::getScore).average().orElse(0.0);
        List<MessageDTO> errorDetectedMessages = messageList.stream().filter(message -> message.getReport() != null).collect(Collectors.toList());
        int errorCount = errorDetectedMessages.size();
        Map<ReportErrorDTO, Long> reportErrorDTOLongMap = errorDetectedMessages.stream().flatMap(messages -> messages.getReport().getErrors().stream()).collect(Collectors.groupingBy(e -> e, Collectors.counting()));
        List<ReportErrorCountDTO> reportErrorCountDTOList = reportErrorDTOLongMap.entrySet().stream().map(e -> {
            ReportErrorCountDTO reportErrorCountDTO = new ReportErrorCountDTO();
            reportErrorCountDTO.setCount(e.getValue());
            reportErrorCountDTO.setCode(e.getKey().getCode());
            reportErrorCountDTO.setDescription(e.getKey().getDescription());
            return reportErrorCountDTO;
        }).collect(Collectors.toList());
        ReportDetailDTO reportDetailDTO = new ReportDetailDTO();
        reportDetailDTO.setMessageCount(messageCount);
        reportDetailDTO.setReportErrorCountDTOList(reportErrorCountDTOList.stream().sorted((e1, e2) -> e2.getCount().compareTo(e1.getCount())).collect(Collectors.toList()));
        reportDetailDTO.setErrorCount(Long.valueOf(errorCount));
        reportDetailDTO.setAverageScore(Double.valueOf(df.format(averageScore)));
        return reportDetailDTO;
    }


}
