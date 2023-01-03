import 'package:chat2learnfe/model/request/RegisterRequest.dart';
import 'package:chat2learnfe/model/response/RegisterResponse.dart';
import 'package:chat2learnfe/navigation/MainNavigation.dart';
import 'package:chat2learnfe/services/AuthenticationClient.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:chat2learnfe/widgets/FadeAnimation.dart';
import 'package:flutter/material.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  AuthenticationClient authenticationClient = AuthenticationClient();

  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _lastnameController = TextEditingController();

  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController =
      TextEditingController();

  @override
  void initState() {
    super.initState();
  }

  void _register() async {
    String name = _nameController.text;
    String lastname = _lastnameController.text;
    String email = _emailController.text;
    String password = _passwordController.text;
    String confirmPassword = _confirmPasswordController.text;

    if (password != confirmPassword) {
      print("Passwords do not match");
      return;
    }
    RegisterRequest request = RegisterRequest(
        name: name,
        lastname: lastname,
        email: email,
        password: password,
        confirmPassword: confirmPassword);
    try {
      RegisterResponse response = await AuthenticationClient.register(request);

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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: Colors.white,
        body: SingleChildScrollView(
          child: Column(
            children: <Widget>[
              Container(
                height: 50,
              ),
              Padding(
                padding: const EdgeInsets.all(30.0),
                child: Column(
                  children: <Widget>[
                    SizedBox(
                      height: 50,
                    ),
                    FadeAnimation(
                      1.8,
                      Container(
                        padding: const EdgeInsets.all(12),
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
                                      bottom: BorderSide(
                                          color: Colors.grey[100]!))),
                              margin: const EdgeInsets.only(top: 10),
                              child: TextField(
                                controller: _nameController,
                                decoration: InputDecoration(
                                    border: InputBorder.none,
                                    hintText: "Name ",
                                    hintStyle:
                                        TextStyle(color: Colors.grey[400])),
                              ),
                            ),
                            Container(
                              padding: const EdgeInsets.all(8.0),
                              decoration: BoxDecoration(
                                  border: Border(
                                      bottom: BorderSide(
                                          color: Colors.grey[100]!))),
                              margin: const EdgeInsets.only(top: 10),
                              child: TextField(
                                controller: _lastnameController,
                                decoration: InputDecoration(
                                    border: InputBorder.none,
                                    hintText: "Lastname ",
                                    hintStyle:
                                        TextStyle(color: Colors.grey[400])),
                              ),
                            ),
                            Container(
                              padding: const EdgeInsets.all(8.0),
                              height: 30,
                            ),
                            Container(
                              padding: const EdgeInsets.all(8.0),
                              decoration: BoxDecoration(
                                  border: Border(
                                      bottom: BorderSide(
                                          color: Colors.grey[100]!))),
                              margin: const EdgeInsets.only(top: 10),
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
                              decoration: BoxDecoration(
                                  border: Border(
                                      bottom: BorderSide(
                                          color: Colors.grey[100]!))),
                              margin: const EdgeInsets.only(top: 10),
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
                            ),
                            Container(
                              padding: const EdgeInsets.all(8.0),
                              margin: const EdgeInsets.only(top: 10),
                              child: TextField(
                                controller: _confirmPasswordController,
                                obscureText: true,
                                enableSuggestions: false,
                                autocorrect: false,
                                decoration: InputDecoration(
                                    border: InputBorder.none,
                                    hintText: "Confirm Password",
                                    hintStyle:
                                        TextStyle(color: Colors.grey[400])),
                              ),
                            )
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(
                      height: 150,
                    ),
                    InkWell(
                      onTap: _register,
                      splashColor: Colors.red,
                      child: FadeAnimation(
                          3,
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
        ));
    ;
  }
}
