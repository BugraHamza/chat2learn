import 'dart:ffi';

import 'package:chat2learnfe/model/dto/ChatBotDTO.dart';
import 'package:chat2learnfe/model/dto/ChatSessionDTO.dart';
import 'package:chat2learnfe/model/response/GetMessagesResponse.dart';
import 'package:chat2learnfe/navigation/MainNavigation.dart';
import 'package:chat2learnfe/pages/ChatBotSelectionPage.dart';
import 'package:chat2learnfe/pages/ChatsPage.dart';
import 'package:chat2learnfe/pages/ReportPage.dart';
import 'package:chat2learnfe/services/ChatBotClient.dart';
import 'package:chat2learnfe/services/ChatSessionClient.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:flutter/material.dart';

class BottomNavigation extends StatefulWidget {
  const BottomNavigation({super.key});

  @override
  State<BottomNavigation> createState() => _BottomNavigationState();
}

class _BottomNavigationState extends State<BottomNavigation> {
  List<ChatBotDTO> _chatBotList = [];
  List<ChatSessionDTO> _chatSessions = [];
  late ChatBotClient _chatBotClient;
  late ChatSessionClient _chatSessionClient;
  int _selectedIndex = 0;

  List<Widget> _widgetOptions(
          List<ChatSessionDTO> chatSessions, onRefresh, updateLastMessage) =>
      <Widget>[
        ChatsPage(
          chatSessions: chatSessions,
          onRefresh: onRefresh,
          updateLastMessage: updateLastMessage,
        ),
        ReportPage()
      ];

  void goToChatBotSelectionPage(
      BuildContext contextList, List<ChatBotDTO> chatBotList) async {
    var result = await Navigator.of(context).push(MaterialPageRoute(
        builder: (context) => ChatBotSelectionPage(
              chatBotList: _chatBotList,
            )));

    if (result != null) {
      setState(() {
        _chatSessions.add(result);
      });
    }
  }

  void _updateLastMessage(int chatSessionId, MessageDTO lastMessage) {
    setState(() {
      _chatSessions
          .firstWhere((element) => element.id == chatSessionId)
          .messages!
          .insert(0, lastMessage);
    });
  }

  // ignore: prefer_final_fields
  List<AppBar> _appBarOptions(context, chatBotList) => <AppBar>[
        AppBar(
          automaticallyImplyLeading: false,
          title: const Text('Chats'),
          actions: [
            IconButton(
              icon: const Icon(Icons.add),
              tooltip: 'Add new chat',
              onPressed: () => goToChatBotSelectionPage(context, chatBotList),
            ),
          ],
        ),
        AppBar(
          automaticallyImplyLeading: false,
          title: const Text('Reports'),
        ),
      ];

  void _onItemTapped(int index) {
    if (index < 2) {
      setState(() {
        _selectedIndex = index;
      });
    } else {
      authenticationNotifiter.setToken("");
      sharedPrefs.name = null;
      Navigator.of(context).pop();
    }
  }

  @override
  void initState() {
    createClients();
    _getChatBots();
    _getChatSessions();
    super.initState();
  }

  void createClients() {
    String token = sharedPrefs.token;
    _chatBotClient = ChatBotClient(token);
    _chatSessionClient = ChatSessionClient(token);
  }

  void _getChatBots() async {
    var chatBotList = await _chatBotClient.getChatBotList();
    setState(() {
      _chatBotList = chatBotList;
    });
  }

  void _getChatSessions() async {
    var chatSessions = await _chatSessionClient.getChatSessionList();
    setState(() {
      _chatSessions = chatSessions;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _appBarOptions(context, _chatBotList).elementAt(_selectedIndex),
      body: _widgetOptions(_chatSessions, _getChatSessions, _updateLastMessage)
          .elementAt(_selectedIndex),
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.chat),
            label: 'Chat',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.report),
            label: 'Report',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.logout),
            label: 'Logout',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.amber[800],
        onTap: _onItemTapped,
      ),
    );
  }
}
