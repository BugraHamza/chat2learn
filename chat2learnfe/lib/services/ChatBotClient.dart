import 'dart:convert';
import 'dart:io';

import 'package:chat2learnfe/model/common/ErrorResponseDTO.dart';
import 'package:chat2learnfe/model/dto/ChatBotDTO.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:http/http.dart' as http;

class ChatBotClient {
  static final baseURL =
      "${sharedPrefs.ip ?? "http://${Platform.isIOS ? "localhost" : "10.0.2.2"}:9092"}/model";

  String token;
  ChatBotClient(this.token);

  Map<String, String> addHeaders() {
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': "Bearer $token"
    };
  }

  Future<List<ChatBotDTO>> getChatBotList() async {
    var url = Uri.parse(baseURL);
    var response = await http.get(url, headers: addHeaders());
    if (response.statusCode == 200) {
      Iterable responseList = json.decode(response.body);
      List<ChatBotDTO> chatBotList = List<ChatBotDTO>.from(
          responseList.map((model) => ChatBotDTO.fromJson(model)));
      return chatBotList;
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }
}
