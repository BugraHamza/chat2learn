import 'dart:convert';
import 'dart:io';

import 'package:chat2learnfe/model/common/ErrorResponseDTO.dart';
import 'package:chat2learnfe/model/request/SendMessageRequest.dart';
import 'package:chat2learnfe/model/response/GetMessagesResponse.dart';
import 'package:chat2learnfe/model/response/SendMessageResponse.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:http/http.dart' as http;

class MessageClient {
  static final baseURL =
      "${sharedPrefs.ip ?? "http://${Platform.isIOS ? "localhost" : "10.0.2.2"}:9092"}/chat";

  String token;
  int chatSessionId;
  MessageClient(this.token, this.chatSessionId);

  Map<String, String> addHeaders() {
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': "Bearer $token"
    };
  }

  Future<GetMessagesResponse> getMessages(int page) async {
    var url = Uri.parse("$baseURL/$chatSessionId/message?page=$page");
    var response = await http.get(url, headers: addHeaders());
    if (response.statusCode == 200) {
      return GetMessagesResponse.fromJson(
          json.decode(utf8.decode(response.bodyBytes)));
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }

  Future<SendMessageResponse> sendMessage(
      SendMessageRequest sendMessageRequest) async {
    var url = Uri.parse("$baseURL/$chatSessionId/message");
    var response = await http.post(url,
        body: jsonEncode(sendMessageRequest.toJson()), headers: addHeaders());
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return SendMessageResponse.fromJson(
          json.decode(utf8.decode(response.bodyBytes)));
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }
}
