class RegisterRequest {
  String? name;
  String? lastname;
  String? email;
  String? password;
  String? confirmPassword;

  RegisterRequest(
      {this.name,
      this.lastname,
      this.email,
      this.password,
      this.confirmPassword});

  RegisterRequest.fromJson(Map<String, dynamic> json) {
    name = json['name'];
    lastname = json['lastname'];
    email = json['email'];
    password = json['password'];
    confirmPassword = json['confirmPassword'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['name'] = this.name;
    data['lastname'] = this.lastname;
    data['email'] = this.email;
    data['password'] = this.password;
    data['confirmPassword'] = this.confirmPassword;
    return data;
  }
}
