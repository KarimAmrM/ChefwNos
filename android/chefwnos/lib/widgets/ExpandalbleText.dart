import 'package:chefwnos/widgets/SmallText.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class ExpandalbleText extends StatefulWidget {
  final String text;
  const ExpandalbleText({ Key? key,required this.text }) : super(key: key);

  @override
  _ExpandalbleTextState createState() => _ExpandalbleTextState();
}

class _ExpandalbleTextState extends State<ExpandalbleText> {
  late String firstHalf;
  late String secondHalf;

  bool hiddenText = true;
  double textHeigth= 100;

  @override
  void initstate(){
    super.initState();
    if(widget.text.length > textHeigth){
      firstHalf = widget.text.substring(0,textHeigth.toInt());
      secondHalf = widget.text.substring(textHeigth.toInt()+1, widget.text.length);
    }else{
      firstHalf = widget.text;
      secondHalf = " ok ";
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      child: secondHalf.isEmpty?SmallText(text: firstHalf):Column(
        children: [
          SmallText(text: hiddenText? (firstHalf + "..."): (firstHalf + secondHalf)),
          InkWell(
            onTap: (){
              setState(() {
                hiddenText = !hiddenText;
              });
            },
            child: Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                SmallText(text: "hiddenText", color: Colors.blue,),
                Icon(Icons.arrow_drop_down, color: Colors.blue,)
              ],
            ),
          ) 
        ],

      ),

    );
  }
}