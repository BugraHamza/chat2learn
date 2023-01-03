import 'package:chat2learnfe/model/notifier/AuthenticationNotifier.dart';
import 'package:chat2learnfe/navigation/BottomNavigation.dart';
import 'package:chat2learnfe/pages/LoginPage.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:flutter/material.dart';

class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

AuthenticationNotifier authenticationNotifiter =
    AuthenticationNotifier(prefs: sharedPrefs);

class _MainNavigationState extends State<MainNavigation> {
  late String _token;
  @override
  void initState() {
    super.initState();
    _token = authenticationNotifiter.getToken();
    authenticationNotifiter.addListener(() => {
          setState(() {
            _token = authenticationNotifiter.getToken();
          })
        });
  }

  @override
  Widget build(BuildContext context) {
    return Navigator(
      pages: [
        MaterialPage(child: LoginPage()),
        if (_token.isNotEmpty) MaterialPage(child: BottomNavigation()),
      ],
      onPopPage: (route, result) => route.didPop(result),
    );
  }
}
