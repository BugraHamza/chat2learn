import 'dart:convert';
import 'dart:io';

import 'package:chat2learnfe/model/common/ErrorResponseDTO.dart';
import 'package:chat2learnfe/model/request/LoginRequest.dart';
import 'package:chat2learnfe/model/request/RegisterRequest.dart';
import 'package:chat2learnfe/model/response/LoginResponse.dart';
import 'package:chat2learnfe/model/response/RegisterResponse.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:http/http.dart' as http;

class AuthenticationClient {
  static final baseURL =
      "${sharedPrefs.ip ?? "http://${Platform.isIOS ? "localhost" : "10.0.2.2"}:9092"}/auth";

  static Map<String, String> addHeaders() {
    Map<String, String> headers = {};
    headers['content-type'] = 'application/json';
    headers['accept'] = 'application/json';
    return headers;
  }

  static Future<LoginResponse> login(LoginRequest loginRequest) async {
    var url = Uri.parse(baseURL);
    print(baseURL);
    var response = await http.post(url,
        body: jsonEncode(loginRequest.toJson()), headers: addHeaders());
    if (response.statusCode == 200) {
      return LoginResponse.fromJson(json.decode(response.body));
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }

  static Future<RegisterResponse> register(
      RegisterRequest registerRequest) async {
    var url = Uri.parse("$baseURL/register");
    var response = await http.post(url,
        body: jsonEncode(registerRequest.toJson()), headers: addHeaders());
    if (response.statusCode == 200) {
      return RegisterResponse.fromJson(json.decode(response.body));
    } else {
      throw Exception(
          ErrorResponseDTO.fromJson(json.decode(response.body)).message);
    }
  }
}
