import 'package:chat2learnfe/model/request/LoginRequest.dart';
import 'package:chat2learnfe/model/response/LoginResponse.dart';
import 'package:chat2learnfe/navigation/MainNavigation.dart';
import 'package:chat2learnfe/pages/RegisterPage.dart';
import 'package:chat2learnfe/services/AuthenticationClient.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:chat2learnfe/widgets/FadeAnimation.dart';
import 'package:flutter/material.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _ipController = TextEditingController();

  void _login() async {
    String email = _emailController.text;
    String password = _passwordController.text;
    print(_emailController.text);
    if (_emailController.text == "ip") {
      showDialog(
          context: context,
          builder: (context) {
            return AlertDialog(
              title: Text('IP DEVELOPER'),
              content: TextField(
                controller: _ipController,
                decoration: InputDecoration(hintText: "IP/ADDRESS"),
              ),
              actions: <Widget>[
                ElevatedButton(
                  child: Text('CANCEL'),
                  onPressed: () {
                    setState(() {
                      Navigator.pop(context);
                    });
                  },
                ),
                ElevatedButton(
                  child: Text('OK'),
                  onPressed: () {
                    setState(() {
                      sharedPrefs.ip = _ipController.text;
                      Navigator.pop(context);
                    });
                  },
                ),
              ],
            );
          });
    } else {
      try {
        LoginResponse response = await AuthenticationClient.login(
            LoginRequest(email: email, password: password));

        authenticationNotifiter.setToken(response.token ?? "");
        sharedPrefs.name = "${response.name} ${response.lastname}";
        Navigator.of(context).pop();
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          backgroundColor: Colors.red,
          content: Text(e.toString()),
        ));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Column(
          children: <Widget>[
            Container(
                height: 400,
                child: FadeAnimation(
                    1.6,
                    Center(
                      child: Text("Chat2Learn",
                          style: TextStyle(
                              fontSize: 40,
                              fontWeight: FontWeight.bold,
                              foreground: Paint()
                                ..shader = LinearGradient(
                                  colors: <Color>[
                                    Color.fromRGBO(143, 148, 251, 1),
                                    Color.fromRGBO(143, 148, 251, .6),
                                    Color.fromARGB(255, 184, 143, 251),
                                    Color.fromARGB(153, 190, 143, 251),
                                    //add more color here.
                                  ],
                                ).createShader(
                                    Rect.fromLTWH(0.0, 0.0, 200.0, 100.0)))),
                    ))),
            Padding(
              padding: const EdgeInsets.all(30.0),
              child: Column(
                children: <Widget>[
                  FadeAnimation(
                    1.8,
                    Container(
                      padding: EdgeInsets.all(5),
                      decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(10),
                          boxShadow: const [
                            BoxShadow(
                                color: Color.fromRGBO(143, 148, 251, .2),
                                blurRadius: 20.0,
                                offset: Offset(0, 10))
                          ]),
                      child: Column(
                        children: <Widget>[
                          Container(
                            padding: const EdgeInsets.all(8.0),
                            decoration: BoxDecoration(
                                border: Border(
                                    bottom:
                                        BorderSide(color: Colors.grey[100]!))),
                            child: TextField(
                              controller: _emailController,
                              decoration: InputDecoration(
                                  border: InputBorder.none,
                                  hintText: "Email ",
                                  hintStyle:
                                      TextStyle(color: Colors.grey[400])),
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.all(8.0),
                            child: TextField(
                              controller: _passwordController,
                              obscureText: true,
                              enableSuggestions: false,
                              autocorrect: false,
                              decoration: InputDecoration(
                                  border: InputBorder.none,
                                  hintText: "Password",
                                  hintStyle:
                                      TextStyle(color: Colors.grey[400])),
                            ),
                          )
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(
                    height: 30,
                  ),
                  InkWell(
                    onTap: _login,
                    splashColor: Colors.red,
                    child: FadeAnimation(
                        2,
                        Container(
                          height: 50,
                          decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(10),
                              gradient: const LinearGradient(colors: [
                                Color.fromRGBO(143, 148, 251, 1),
                                Color.fromRGBO(143, 148, 251, .6),
                              ])),
                          child: const Center(
                            child: Text(
                              "Login",
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold),
                            ),
                          ),
                        )),
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  InkWell(
                    onTap: () => {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => const RegisterPage()),
                      )
                    },
                    splashColor: Colors.red,
                    child: FadeAnimation(
                        2,
                        Container(
                          height: 50,
                          decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(10),
                              gradient: const LinearGradient(colors: [
                                Color.fromARGB(255, 184, 143, 251),
                                Color.fromARGB(153, 190, 143, 251),
                              ])),
                          child: const Center(
                            child: Text(
                              "Register",
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold),
                            ),
                          ),
                        )),
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}
