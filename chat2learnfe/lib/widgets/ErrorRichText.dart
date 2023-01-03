import 'package:flutter/material.dart';
import 'package:flutter/src/widgets/container.dart';
import 'package:flutter/src/widgets/framework.dart';

final regexToSplitTags = RegExp(r"(<\w[^<]*<\/\w>)");
final regexToFindEdit = RegExp("edit='(.*?)'");
final regexToFindDefault = RegExp(r"('>(.*?)<\/)");
final regexToFindErrorType = RegExp(r"type='(.*?)'");
final regexToFindDescritption = RegExp(r"description='(.*?)'");

class ErrorRichText extends StatelessWidget {
  final String taggedText;
  final bool colorizeCorrect;

  const ErrorRichText(
      {required this.taggedText, this.colorizeCorrect = false, super.key});

  @override
  Widget build(BuildContext context) {
    generateSpans(taggedText);
    return GestureDetector(
      onLongPress: () => print(taggedText),
      child: RichText(
        text: TextSpan(
          // set the default style for the children TextSpans
          style: Theme.of(context).textTheme.bodyMedium,
          children: generateSpans(taggedText),
        ),
      ),
    );
  }

  List<TextSpan> generateSpans(String taggedText) {
    List<TextSpan> spans = [];
    final regexList = regexToSplitTags.allMatches(taggedText);
    List<String> splitTagsList = [];
    print(taggedText);
    print("------------------");
    int stringIndex = 0;
    for (final tag in regexList) {
      if (stringIndex < tag.start) {
        splitTagsList.add(taggedText.substring(stringIndex, tag.start));
        stringIndex = tag.start;
      }
      if (tag.start == stringIndex) {
        splitTagsList.add(taggedText.substring(tag.start, tag.end));
        stringIndex = tag.end;
      }
    }
    if (regexList.last.end < taggedText.length) {
      splitTagsList
          .add(taggedText.substring(regexList.last.end, taggedText.length));
    }
    for (final split in splitTagsList) {
      if (split.startsWith("<")) {
        final edit = regexToFindEdit.firstMatch(split);
        final defaultText = regexToFindDefault.firstMatch(split);
        final errorType = regexToFindErrorType.firstMatch(split);
        final description = regexToFindDescritption.firstMatch(split);
        if (edit != null && defaultText != null && errorType != null) {
          spans.add(TextSpan(
              text: !colorizeCorrect ? edit.group(1) : defaultText.group(2),
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  color: !colorizeCorrect ? Colors.green : Colors.red)));
        } else {
          spans.add(TextSpan(text: split, style: TextStyle(fontSize: 13)));
        }
      } else {
        spans.add(TextSpan(text: split, style: TextStyle(fontSize: 13)));
      }
    }
    return spans;
  }
}
