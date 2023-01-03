import 'dart:ffi';

import 'package:chat2learnfe/model/dto/ChatSessionDTO.dart';
import 'package:chat2learnfe/model/response/GetMessagesResponse.dart';
import 'package:chat2learnfe/pages/ChatDetailPage.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class ChatsPage extends StatefulWidget {
  List<ChatSessionDTO> chatSessions;
  var updateLastMessage;
  var onRefresh;
  ChatsPage(
      {super.key,
      required this.chatSessions,
      required this.onRefresh,
      required this.updateLastMessage});

  @override
  State<ChatsPage> createState() => _ChatsPageState();
}

class _ChatsPageState extends State<ChatsPage> {
  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: () async {
        widget.onRefresh();
        return Future.value();
      },
      child: ListView.separated(
          separatorBuilder: (context, index) => const Divider(
                color: Color.fromARGB(255, 134, 134, 134),
                height: 1,
                thickness: 0.3,
              ),
          itemCount: widget.chatSessions.length,
          itemBuilder: (BuildContext context, int index) {
            return ListTile(
              onTap: () async {
                MessageDTO? result = await Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => ChatDetailPage(
                      chatSession: widget.chatSessions[index],
                    ),
                  ),
                );
                if (result != null) {
                  widget.updateLastMessage(
                      widget.chatSessions[index].id, result);
                }
              },
              leading: CircleAvatar(
                radius: 25,
                backgroundImage: NetworkImage(
                    "https://robohash.org/${widget.chatSessions[index].model?.name}"),
              ),
              trailing: const Icon(Icons.chat_bubble_outline),
              title: Text("${widget.chatSessions[index].model?.name}"),
              subtitle: Text(widget.chatSessions[index].messages!.isNotEmpty
                  ? "${DateFormat('dd/MM/yyyy').format(DateTime.parse(widget.chatSessions[index].messages![0].createdDate!))} - ${widget.chatSessions[index].messages![0].text}"
                  : "No messages yet"),
            );
          }),
    );
  }
}
