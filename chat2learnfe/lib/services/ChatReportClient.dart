import 'dart:convert';
import 'dart:io';

import 'package:chat2learnfe/model/common/ErrorResponseDTO.dart';
import 'package:chat2learnfe/model/dto/ChatBotDTO.dart';
import 'package:chat2learnfe/model/response/ChatSessionReportResponse.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:http/http.dart' as http;

class ChatReportClient {
  static final baseURL =
      "${sharedPrefs.ip ?? "http://${Platform.isIOS ? "localhost" : "10.0.2.2"}:9092"}/report";

  String token;
  int chatSessionId;
  ChatReportClient(this.token, this.chatSessionId);

  Map<String, String> addHeaders() {
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': "Bearer $token"
    };
  }

  Future<ChatSessionReportResponse> getChatSessionReport() async {
    var url = Uri.parse("$baseURL/$chatSessionId");
    var response = await http.get(url, headers: addHeaders());
    if (response.statusCode == 200) {
      return ChatSessionReportResponse.fromJson(json.decode(response.body));
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }
}
