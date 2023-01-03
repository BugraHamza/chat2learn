class ChatSessionReportResponse {
  int? messageCount;
  int? errorCount;
  double? averageScore;

  ChatSessionReportResponse(
      {this.messageCount, this.errorCount, this.averageScore});

  ChatSessionReportResponse.fromJson(Map<String, dynamic> json) {
    messageCount = json['messageCount'];
    errorCount = json['errorCount'];
    averageScore = json['averageScore'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['messageCount'] = this.messageCount;
    data['errorCount'] = this.errorCount;
    data['avarageScore'] = this.averageScore;
    return data;
  }
}
