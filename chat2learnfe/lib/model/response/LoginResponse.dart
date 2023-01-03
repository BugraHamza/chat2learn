class LoginResponse {
  int? id;
  String? name;
  String? lastname;
  String? email;
  String? token;

  LoginResponse({this.id, this.name, this.lastname, this.email, this.token});

  LoginResponse.fromJson(Map<String, dynamic> json) {
    id = json['id'];
    name = json['name'];
    lastname = json['lastname'];
    email = json['email'];
    token = json['token'];
  }

  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = new Map<String, dynamic>();
    data['id'] = this.id;
    data['name'] = this.name;
    data['lastname'] = this.lastname;
    data['email'] = this.email;
    data['token'] = this.token;
    return data;
  }
}
