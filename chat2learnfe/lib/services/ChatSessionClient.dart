import 'dart:convert';
import 'dart:io';

import 'package:chat2learnfe/model/common/ErrorResponseDTO.dart';
import 'package:chat2learnfe/model/dto/ChatSessionDTO.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:http/http.dart' as http;

class ChatSessionClient {
  static final baseURL =
      "${sharedPrefs.ip ?? "http://${Platform.isIOS ? "localhost" : "10.0.2.2"}:9092"}/chat";

  String token;
  ChatSessionClient(this.token);

  Map<String, String> addHeaders() {
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': "Bearer $token"
    };
  }

  Future<List<ChatSessionDTO>> getChatSessionList() async {
    var url = Uri.parse(baseURL);
    var response = await http.get(url, headers: addHeaders());
    if (response.statusCode == 200) {
      Iterable responseList = json.decode(response.body);
      List<ChatSessionDTO> chatSessionList = List<ChatSessionDTO>.from(
          responseList.map((model) => ChatSessionDTO.fromJson(model)));
      return chatSessionList;
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }

  Future<ChatSessionDTO> create(int modelId) async {
    var url = Uri.parse("$baseURL/$modelId");
    var response = await http.post(url, headers: addHeaders());
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return ChatSessionDTO.fromJson(json.decode(response.body));
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }
}
