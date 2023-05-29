import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ChatPage extends StatefulWidget {
  @override
  _ChatPageState createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  List<Message> messages = [];
  TextEditingController textController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Chat Page'),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: messages.length,
              itemBuilder: (context, index) {
                final message = messages[index];
                return ListTile(
                  title: Text(message.text),
                  subtitle: Text(message.isSentByUser ? 'You' : 'Bot'),
                  tileColor: message.isSentByUser
                      ? Colors.blue[100]
                      : Colors.grey[300],
                  contentPadding:
                      EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                );
              },
            ),
          ),
          Container(
            padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: textController,
                    decoration: InputDecoration(
                      hintText: 'Type your message...',
                    ),
                  ),
                ),
                IconButton(
                  onPressed: () {
                    final userMessage = textController.text;
                    sendMessage(userMessage);
                  },
                  icon: Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Future<void> sendMessage(String userMessage) async {
    setState(() {
      messages.add(Message(text: userMessage, isSentByUser: true));
      textController.clear();
    });

    final botResponse = await getBotResponse(userMessage);
    setState(() {
      messages.add(Message(text: botResponse, isSentByUser: false));
    });
  }

  Future<String> getBotResponse(String userMessage) async {
    // TODO: Make API request to get bot response
    // Replace the API_URL with the actual URL of your API
    final response = await http.get(Uri.parse('API_URL?message=$userMessage'));

    if (response.statusCode == 200) {
      final jsonResponse = json.decode(response.body);
      final botResponse = jsonResponse['response'] as String;
      return botResponse;
    } else {
      throw Exception('Failed to get bot response');
    }
  }
}

class Message {
  final String text;
  final bool isSentByUser;

  Message({required this.text, required this.isSentByUser});
}
