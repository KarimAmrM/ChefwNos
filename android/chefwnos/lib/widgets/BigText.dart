import 'package:chefwnos/utils/colors.dart';
import 'package:flutter/cupertino.dart';

class BigText extends StatelessWidget {

  Color? color;
  final String text;
  double size;
  TextOverflow overFlow;


  BigText({ Key? key,  
  this.color=const Color.fromARGB(255, 29, 29, 29) , 
  required this.text,
  this.size = 20,
  this.overFlow = TextOverflow.visible,
   }) : super(key: key);

  @override
  Widget build(BuildContext context){
    return Text(
      text,
      overflow: overFlow,
      style: TextStyle(
        color: color,
        fontSize: size,
        fontWeight: FontWeight.bold,
      ),

    );
  }
}