import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:chat2learnfe/model/common/ErrorResponseDTO.dart';
import 'package:chat2learnfe/model/dto/ReportDetailDTO.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:http/http.dart' as http;

class ReportClient {
  static final baseURL =
      "${sharedPrefs.ip ?? "http://${Platform.isIOS ? "localhost" : "10.0.2.2"}:9092"}/report";

  String token;
  ReportClient(this.token);

  Map<String, String> addHeaders() {
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': "Bearer $token"
    };
  }

  Future<ReportDetailDTO> getReport() async {
    var url = Uri.parse(baseURL);
    var response = await http.get(url, headers: addHeaders());
    if (response.statusCode == 200) {
      return ReportDetailDTO.fromJson(
          json.decode(utf8.decode(response.bodyBytes)));
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }
}
