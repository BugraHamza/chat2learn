class ReportDetailDTO {
  int? messageCount;
  int? errorCount;
  double? averageScore;
  List<ReportErrorCountDTOList>? reportErrorCountDTOList;
  Map<DateTime, double>? scoreMap;

  ReportDetailDTO(
      {this.messageCount,
      this.errorCount,
      this.averageScore,
      this.reportErrorCountDTOList,
      this.scoreMap});

  ReportDetailDTO.fromJson(Map<String, dynamic> json) {
    messageCount = json['messageCount'];
    errorCount = json['errorCount'];
    averageScore = json['averageScore'];
    if (json['reportErrorCountDTOList'] != null) {
      reportErrorCountDTOList = <ReportErrorCountDTOList>[];
      json['reportErrorCountDTOList'].forEach((v) {
        reportErrorCountDTOList!.add(new ReportErrorCountDTOList.fromJson(v));
      });
    }
    json['scoreMap'] =
        json['scoreMap'].map((k, v) => MapEntry(DateTime.parse(k), v));
    scoreMap = json['scoreMap'] != null
        ? new Map<DateTime, double>.from(json['scoreMap'])
        : null;
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['messageCount'] = this.messageCount;
    data['errorCount'] = this.errorCount;
    data['averageScore'] = this.averageScore;
    if (this.reportErrorCountDTOList != null) {
      data['reportErrorCountDTOList'] =
          this.reportErrorCountDTOList!.map((v) => v.toJson()).toList();
    }
    return data;
  }
}

class ReportErrorCountDTOList {
  String? code;
  String? description;
  int? count;

  ReportErrorCountDTOList({this.code, this.description, this.count});

  ReportErrorCountDTOList.fromJson(Map<String, dynamic> json) {
    code = json['code'];
    description = json['description'];
    count = json['count'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['code'] = this.code;
    data['description'] = this.description;
    data['count'] = this.count;
    return data;
  }
}
