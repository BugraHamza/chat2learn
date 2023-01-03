// ignore: file_names
import 'package:shared_preferences/shared_preferences.dart';

class SharedPrefs {
  static SharedPreferences? _sharedPrefs;

  init() async {
    _sharedPrefs ??= await SharedPreferences.getInstance();
  }

  String get token {
    return _sharedPrefs?.getString("token") ?? "";
  }

  set token(String? token) {
    if (token == null) {
      return;
    }
    _sharedPrefs?.setString("token", token);
  }

  set name(String? name) {
    if (name == null) {
      return;
    }
    _sharedPrefs?.setString("name", name);
  }

  String? get name {
    return _sharedPrefs?.getString("name");
  }

  set ip(String? ip) {
    if (ip == null) {
      return;
    }
    _sharedPrefs?.setString("ip", ip);
  }

  String? get ip {
    return _sharedPrefs?.getString("ip");
  }
}

final sharedPrefs = SharedPrefs();
