import 'package:chat2learnfe/model/response/GetMessagesResponse.dart';
import 'package:chat2learnfe/widgets/ErrorRichText.dart';
import 'package:flutter/material.dart';

final regexToSplitTags = RegExp(r"(<\w[^<]*<\/\w>)");
final regexToFindEdit = RegExp("edit='(.*?)'");
final regexToFindDefault = RegExp(r"('>(.*?)<\/)");
final regexToFindErrorType = RegExp(r"type='(.*?)'");
final regexToFindDescritption = RegExp(r"description='(.*?)'");

class AnimatedMessage extends StatefulWidget {
  MessageDTO message;
  final Key key;
  AnimatedMessage({required this.message, required this.key}) : super(key: key);

  @override
  State<AnimatedMessage> createState() => _AnimatedMessageState();
}

class _AnimatedMessageState extends State<AnimatedMessage> {
  void _showErrors() {
    if (widget.message.report == null) return;
    final regexList = regexToSplitTags
        .allMatches(widget.message.report!.taggedCorrectText!)
        .toList();

    showModalBottomSheet<void>(
      context: context,
      builder: (BuildContext context) {
        return Container(
          height: 300,
          child: ListView.separated(
              itemBuilder: ((context, index) {
                var tagReg = regexList[index];
                var tag = widget.message.report!.taggedCorrectText!
                    .substring(tagReg.start, tagReg.end);
                final edit = regexToFindEdit.firstMatch(tag);
                final defaultText = regexToFindDefault.firstMatch(tag);
                final errorType = regexToFindErrorType.firstMatch(tag);
                final description = regexToFindDescritption.firstMatch(tag);
                return ListTile(
                  title: Text(errorType!.group(1)!),
                  subtitle: Text(description!.group(1)!),
                  trailing: FittedBox(
                    fit: BoxFit.fill,
                    child: Row(
                      children: <Widget>[
                        Text(
                          defaultText!.group(2)!,
                          style: TextStyle(color: Colors.red),
                        ),
                        Icon(Icons.arrow_right_sharp),
                        Text(
                          edit!.group(1)!,
                          style: TextStyle(color: Colors.green),
                        ),
                      ],
                    ),
                  ),
                );
              }),
              separatorBuilder: ((context, index) => Divider()),
              itemCount: regexList.length),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onLongPress: _showErrors,
      child: Container(
        padding:
            const EdgeInsets.only(left: 16, right: 16, top: 10, bottom: 10),
        child: Align(
          alignment: Alignment.centerRight,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              color: widget.message.report == null
                  ? const Color.fromARGB(255, 225, 255, 199)
                  : (widget.message.score == null
                              ? 1.0
                              : widget.message.score!) <
                          0.2
                      ? Color.fromARGB(255, 249, 101, 101)
                      : Color.fromARGB(255, 249, 198, 198),
            ),
            padding: const EdgeInsets.all(16),
            child: Column(children: [
              Text(
                widget.message.text!,
              ),
              if (widget.message.report != null)
                Padding(
                    padding: const EdgeInsets.only(top: 10),
                    child: ErrorRichText(
                        taggedText: widget.message.report!.taggedCorrectText!))
            ]),
          ),
        ),
      ),
    );
  }
}
