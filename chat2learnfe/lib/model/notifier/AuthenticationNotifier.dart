import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:flutter/material.dart';

class AuthenticationNotifier extends ChangeNotifier {
  final SharedPrefs prefs;
  late String _token;

  AuthenticationNotifier({required this.prefs});

  void _setPrefItems() {
    prefs.token = _token;
  }

  void _getPrefItems() {
    _token = prefs.token;
  }

  void setToken(String token) {
    _token = token;
    _setPrefItems();
    notifyListeners();
  }

  String getToken() {
    _getPrefItems();
    return _token;
  }
}
