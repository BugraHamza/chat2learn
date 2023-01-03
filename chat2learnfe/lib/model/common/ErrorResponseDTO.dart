class ErrorResponseDTO {
  String? message;
  String? httpStatus;
  String? timestamp;

  ErrorResponseDTO({this.message, this.httpStatus, this.timestamp});

  ErrorResponseDTO.fromJson(Map<String, dynamic> json) {
    message = json['message'];
    httpStatus = json['httpStatus'];
    timestamp = json['timestamp'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['message'] = this.message;
    data['httpStatus'] = this.httpStatus;
    data['timestamp'] = this.timestamp;
    return data;
  }
}
