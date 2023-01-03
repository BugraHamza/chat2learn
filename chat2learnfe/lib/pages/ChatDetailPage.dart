import 'dart:ffi';

import 'package:chat2learnfe/model/dto/ChatSessionDTO.dart';
import 'package:chat2learnfe/model/request/SendMessageRequest.dart';
import 'package:chat2learnfe/model/response/ChatSessionReportResponse.dart';
import 'package:chat2learnfe/model/response/GetMessagesResponse.dart';
import 'package:chat2learnfe/model/response/SendMessageResponse.dart';
import 'package:chat2learnfe/services/ChatReportClient.dart';
import 'package:chat2learnfe/services/MessageClient.dart';
import 'package:chat2learnfe/services/SharedPrefs.dart';
import 'package:chat2learnfe/widgets/AnimatedMessage.dart';
import 'package:flutter/material.dart';

class ChatDetailPage extends StatefulWidget {
  ChatSessionDTO chatSession;
  ChatDetailPage({super.key, required this.chatSession});

  @override
  State<ChatDetailPage> createState() => _ChatDetailPageState();
}

class _ChatDetailPageState extends State<ChatDetailPage> {
  List<MessageDTO> _messages = [];
  int page = 0;
  late int _totalPages;
  bool _isLoading = true;
  bool _loadingMore = false;
  bool _chatbotTyping = false;
  late bool _isLastPage;
  final ScrollController _scrollController = ScrollController();
  final TextEditingController _messageEditingController =
      TextEditingController();
  late MessageClient _messageClient;
  late ChatReportClient _chatReportClient;

  void sendMessage() async {
    if (_messageEditingController.text.isNotEmpty) {
      MessageDTO newMessage = MessageDTO(
        text: _messageEditingController.text,
        senderType: "USER",
      );
      setState(() {
        _messageEditingController.clear();
        _chatbotTyping = true;
        _messages.insert(0, newMessage);
      });
      try {
        SendMessageResponse response = await _messageClient
            .sendMessage(SendMessageRequest(message: newMessage.text));
        setState(() {
          _chatbotTyping = false;
          _messages[0].report = response.personMessage?.report;
          _messages.insert(0, response.botMessage!);
        });
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          backgroundColor: Colors.red,
          content: Text(e.toString()),
        ));
        _messages.remove(newMessage);
      }
    }
  }

  @override
  void initState() {
    super.initState();
    createClients();
    getMessages();
    _scrollController.addListener(_scrollListener);
  }

  void createClients() {
    String token = sharedPrefs.token;
    _messageClient = MessageClient(token, widget.chatSession.id!);
    _chatReportClient = ChatReportClient(token, widget.chatSession.id!);
  }

  void getMessages() async {
    if (page > 0) {
      setState(() {
        _loadingMore = true;
      });
    }
    GetMessagesResponse response = await _messageClient.getMessages(page);
    if (response.content == null) {
      return;
    }
    setState(() {
      List<MessageDTO> messages = response.content!;
      _totalPages = response.totalPages!;
      _isLastPage = response.last!;
      _messages.addAll(messages);
      if (page == 0) {
        _isLoading = false;
      } else {
        _loadingMore = false;
      }
    });
  }

  @override
  void dispose() {
    _scrollController.removeListener(_scrollListener);
    _scrollController.dispose();
    super.dispose();
  }

  void _showReport() async {
    ChatSessionReportResponse response =
        await _chatReportClient.getChatSessionReport();
    double errorRate = 0.00;
    if (response.messageCount != 0) {
      errorRate = response.errorCount! / response.messageCount!;
    }
    showDialog(
        context: context,
        builder: (context) {
          return Dialog(
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8)),
              elevation: 16,
              child: Container(
                height: 100,
                padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    Row(
                      children: [
                        Text("Avarage Score :"),
                        Spacer(),
                        Text(
                          "${response.averageScore! * 100} %",
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Color.lerp(Colors.red, Colors.green,
                                response.averageScore!),
                          ),
                        )
                      ],
                    ),
                    SizedBox(height: 16),
                    Row(
                      children: [
                        Text("Error Rate :"),
                        Spacer(),
                        Text(
                          "${(errorRate * 100).toStringAsFixed(1)} %",
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color:
                                Color.lerp(Colors.green, Colors.red, errorRate),
                          ),
                        )
                      ],
                    ),
                  ],
                ),
              ));
        });
  }

  Widget buildItem(int index, MessageDTO message) {
    if (message.senderType == "BOT") {
      return Container(
        key: Key('message-${message.id}'),
        padding:
            const EdgeInsets.only(left: 16, right: 16, top: 10, bottom: 10),
        child: Align(
          alignment: Alignment.centerLeft,
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              color: const Color.fromARGB(255, 220, 220, 220),
            ),
            padding: const EdgeInsets.all(16),
            child: Text(message.text!),
          ),
        ),
      );
    } else {
      return AnimatedMessage(
        message: message,
        key: Key('message-${message.id}'),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        automaticallyImplyLeading: false,
        backgroundColor: Colors.white,
        flexibleSpace: SafeArea(
          child: Container(
            padding: const EdgeInsets.only(right: 16),
            child: Row(
              children: <Widget>[
                IconButton(
                  onPressed: () {
                    Navigator.of(context)
                        .pop(_messages.isNotEmpty ? _messages.first : null);
                  },
                  icon: const Icon(
                    Icons.arrow_back,
                    color: Colors.black,
                  ),
                ),
                const SizedBox(
                  width: 2,
                ),
                CircleAvatar(
                  backgroundImage: NetworkImage(
                      "https://robohash.org/${widget.chatSession.model?.name}"),
                  maxRadius: 20,
                ),
                const SizedBox(
                  width: 12,
                ),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      Text(
                        widget.chatSession.model?.name! ?? "Chatbot",
                        style: const TextStyle(
                            fontSize: 16, fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(
                        height: 6,
                      ),
                      Text(
                        _chatbotTyping ? "Typing ..." : "Online",
                        style: TextStyle(
                            color: Colors.grey.shade600, fontSize: 13),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  onPressed: _showReport,
                  icon: const Icon(Icons.report),
                ),
              ],
            ),
          ),
        ),
      ),
      body: Column(
        children: <Widget>[
          if (_loadingMore)
            Container(
              margin: EdgeInsets.only(top: 10),
              height: 10,
              width: 10,
              child: const CircularProgressIndicator(),
            ),
          Expanded(
            child: _isLoading
                ? const Center(
                    child: CircularProgressIndicator(
                      value: null,
                      strokeWidth: 7.0,
                    ),
                  )
                : _messages.isEmpty
                    ? const Center(
                        child: Text(
                          "No messages",
                          style: TextStyle(
                              color: Colors.black54,
                              fontSize: 16,
                              fontWeight: FontWeight.w600),
                        ),
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.all(10.0),
                        itemCount: _messages.length,
                        reverse: true,
                        controller: _scrollController,
                        itemBuilder: (BuildContext context, int index) {
                          return buildItem(index, _messages[index]);
                        },
                      ),
          ),
          Container(
            padding: const EdgeInsets.only(bottom: 20),
            decoration: const BoxDecoration(
              boxShadow: [
                BoxShadow(
                  color: Color.fromARGB(255, 56, 56, 56),
                  blurRadius: 1.0,
                  spreadRadius: 0.0,
                  offset: Offset(1.0, 1.0), // shadow direction: bottom right
                )
              ],
              color: Colors.white,
              borderRadius: BorderRadius.only(
                topLeft: Radius.circular(8),
                topRight: Radius.circular(8),
              ),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: <Widget>[
                Expanded(
                  child: Container(
                    margin: const EdgeInsets.only(top: 10, left: 20),
                    padding: const EdgeInsets.only(left: 10),
                    decoration: const BoxDecoration(
                      color: Color.fromARGB(255, 237, 237, 237),
                      borderRadius: BorderRadius.all(Radius.circular(4)),
                    ),
                    child: TextField(
                      autocorrect: false,
                      controller: _messageEditingController,
                      maxLength: 128,
                      maxLines: null,
                      decoration: const InputDecoration(
                        hintText: "Write message...",
                        hintStyle: TextStyle(color: Colors.black54),
                        border: InputBorder.none,
                      ),
                    ),
                  ),
                ),
                ValueListenableBuilder(
                  valueListenable: _messageEditingController,
                  builder: (context, value, child) => IconButton(
                    iconSize: 14,
                    onPressed: value.text.isNotEmpty ? sendMessage : null,
                    disabledColor: Colors.grey,
                    color: Colors.blue,
                    icon: const Icon(
                      Icons.send,
                    ),
                  ),
                )
              ],
            ),
          ),
        ],
      ),
    );
    ;
  }

  void _scrollListener() {
    if (_scrollController.position.extentAfter == 0) {
      if (!_isLastPage && !_loadingMore && page + 1 <= _totalPages) {
        page++;
        getMessages();
      }
    }
  }
}
