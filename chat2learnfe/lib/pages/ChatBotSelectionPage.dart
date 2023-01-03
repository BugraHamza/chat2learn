import 'package:chat2learnfe/model/dto/ChatBotDTO.dart';
import 'package:chat2learnfe/services/ChatBotClient.dart';
import 'package:chat2learnfe/services/ChatSessionClient.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:flutter/material.dart';
import 'package:flutter/src/widgets/container.dart';
import 'package:flutter/src/widgets/framework.dart';

class ChatBotSelectionPage extends StatefulWidget {
  List<ChatBotDTO> chatBotList = [];
  ChatBotSelectionPage({super.key, required this.chatBotList});

  @override
  State<ChatBotSelectionPage> createState() => _ChatBotSelectionPageState();
}

class _ChatBotSelectionPageState extends State<ChatBotSelectionPage>
    with SingleTickerProviderStateMixin {
  late ChatSessionClient _chatSessionClient;
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController =
        TabController(vsync: this, length: widget.chatBotList.length);
    createClients();
  }

  void createClients() {
    String token = sharedPrefs.token;
    _chatSessionClient = ChatSessionClient(token);
  }

  void createChatSession(int chatBotId) async {
    try {
      var chatSession = await _chatSessionClient.create(chatBotId);
      Navigator.of(context).pop(chatSession);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        backgroundColor: Colors.red,
        content: Text(e.toString()),
      ));
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chat Bot Selection'),
      ),
      body: TabBarView(
        controller: _tabController,
        children: widget.chatBotList.map((ChatBotDTO chatBotDTO) {
          final String label = chatBotDTO.name ?? 'No name';
          return Column(children: [
            const SizedBox(
              height: 30,
            ),
            CircleAvatar(
              backgroundImage: NetworkImage("https://robohash.org/$label"),
              maxRadius: 100,
            ),
            const SizedBox(
              height: 30,
            ),
            Expanded(
                child: Container(
              child: Text(chatBotDTO.description ?? 'No description'),
            )),
            FittedBox(
              fit: BoxFit.contain,
              child: Container(
                  height: 60,
                  margin: EdgeInsets.symmetric(horizontal: 20),
                  child: ElevatedButton(
                    onPressed: () {
                      if (chatBotDTO.id == null) {
                        return;
                      }
                      createChatSession(chatBotDTO.id!);
                    },
                    child: Text(
                      maxLines: 1,
                      'Start Chatting with $label',
                      style: const TextStyle(fontSize: 24),
                    ),
                  )),
            ),
            SizedBox(height: 80),
          ]);
        }).toList(),
      ),
    );
  }
}
