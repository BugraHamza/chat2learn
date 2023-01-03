import 'package:chat2learnfe/model/response/GetMessagesResponse.dart';

class SendMessageResponse {
  MessageDTO? personMessage;
  MessageDTO? botMessage;

  SendMessageResponse({this.personMessage, this.botMessage});

  SendMessageResponse.fromJson(Map<String, dynamic> json) {
    personMessage = json['personMessage'] != null
        ? new MessageDTO.fromJson(json['personMessage'])
        : null;
    botMessage = json['botMessage'] != null
        ? new MessageDTO.fromJson(json['botMessage'])
        : null;
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    if (this.personMessage != null) {
      data['personMessage'] = this.personMessage!.toJson();
    }
    if (this.botMessage != null) {
      data['botMessage'] = this.botMessage!.toJson();
    }
    return data;
  }
}

class PersonMessage {
  String? createdBy;
  String? createdDate;
  String? lastModifiedBy;
  String? lastModifiedDate;
  int? id;
  String? text;
  String? senderType;
  Report? report;

  PersonMessage(
      {this.createdBy,
      this.createdDate,
      this.lastModifiedBy,
      this.lastModifiedDate,
      this.id,
      this.text,
      this.senderType,
      this.report});

  PersonMessage.fromJson(Map<String, dynamic> json) {
    createdBy = json['createdBy'];
    createdDate = json['createdDate'];
    lastModifiedBy = json['lastModifiedBy'];
    lastModifiedDate = json['lastModifiedDate'];
    id = json['id'];
    text = json['text'];
    senderType = json['senderType'];
    report =
        json['report'] != null ? new Report.fromJson(json['report']) : null;
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['createdBy'] = this.createdBy;
    data['createdDate'] = this.createdDate;
    data['lastModifiedBy'] = this.lastModifiedBy;
    data['lastModifiedDate'] = this.lastModifiedDate;
    data['id'] = this.id;
    data['text'] = this.text;
    data['senderType'] = this.senderType;
    if (this.report != null) {
      data['report'] = this.report!.toJson();
    }
    return data;
  }
}

class Report {
  String? createdBy;
  String? createdDate;
  String? lastModifiedBy;
  String? lastModifiedDate;
  int? id;
  String? correctText;
  String? taggedCorrectText;
  List<Errors>? errors;

  Report(
      {this.createdBy,
      this.createdDate,
      this.lastModifiedBy,
      this.lastModifiedDate,
      this.id,
      this.correctText,
      this.taggedCorrectText,
      this.errors});

  Report.fromJson(Map<String, dynamic> json) {
    createdBy = json['createdBy'];
    createdDate = json['createdDate'];
    lastModifiedBy = json['lastModifiedBy'];
    lastModifiedDate = json['lastModifiedDate'];
    id = json['id'];
    correctText = json['correctText'];
    taggedCorrectText = json['taggedCorrectText'];
    if (json['errors'] != null) {
      errors = <Errors>[];
      json['errors'].forEach((v) {
        errors!.add(new Errors.fromJson(v));
      });
    }
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['createdBy'] = this.createdBy;
    data['createdDate'] = this.createdDate;
    data['lastModifiedBy'] = this.lastModifiedBy;
    data['lastModifiedDate'] = this.lastModifiedDate;
    data['id'] = this.id;
    data['correctText'] = this.correctText;
    data['taggedCorrectText'] = this.taggedCorrectText;
    if (this.errors != null) {
      data['errors'] = this.errors!.map((v) => v.toJson()).toList();
    }
    return data;
  }
}

class Errors {
  String? createdBy;
  String? createdDate;
  String? lastModifiedBy;
  String? lastModifiedDate;
  String? code;

  Errors(
      {this.createdBy,
      this.createdDate,
      this.lastModifiedBy,
      this.lastModifiedDate,
      this.code});

  Errors.fromJson(Map<String, dynamic> json) {
    createdBy = json['createdBy'];
    createdDate = json['createdDate'];
    lastModifiedBy = json['lastModifiedBy'];
    lastModifiedDate = json['lastModifiedDate'];
    code = json['code'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['createdBy'] = this.createdBy;
    data['createdDate'] = this.createdDate;
    data['lastModifiedBy'] = this.lastModifiedBy;
    data['lastModifiedDate'] = this.lastModifiedDate;
    data['code'] = this.code;
    return data;
  }
}

class BotMessage {
  String? createdBy;
  String? createdDate;
  String? lastModifiedBy;
  String? lastModifiedDate;
  int? id;
  String? text;
  String? senderType;
  Null? report;

  BotMessage(
      {this.createdBy,
      this.createdDate,
      this.lastModifiedBy,
      this.lastModifiedDate,
      this.id,
      this.text,
      this.senderType,
      this.report});

  BotMessage.fromJson(Map<String, dynamic> json) {
    createdBy = json['createdBy'];
    createdDate = json['createdDate'];
    lastModifiedBy = json['lastModifiedBy'];
    lastModifiedDate = json['lastModifiedDate'];
    id = json['id'];
    text = json['text'];
    senderType = json['senderType'];
    report = json['report'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['createdBy'] = this.createdBy;
    data['createdDate'] = this.createdDate;
    data['lastModifiedBy'] = this.lastModifiedBy;
    data['lastModifiedDate'] = this.lastModifiedDate;
    data['id'] = this.id;
    data['text'] = this.text;
    data['senderType'] = this.senderType;
    data['report'] = this.report;
    return data;
  }
}
