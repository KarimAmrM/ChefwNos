import 'package:chefwnos/pages/Home/foodPageBody.dart';
import 'package:chefwnos/utils/colors.dart';
import 'package:chefwnos/widgets/BigText.dart';
import 'package:chefwnos/widgets/SmallText.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class MainPage extends StatefulWidget {
  const MainPage({ Key? key }) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _MainPageState createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Container(
          child: Container(
            margin: EdgeInsets.only(top: 50, left: 20, right: 20),
            child:Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  children: [
                    BigText(text: "Hello,"),
                    SmallText(text:"there")
                  ],
                ),
                Container(
                  width: 45,
                  height: 45,
                  child: Icon(Icons.search, color: Colors.white,),
                  decoration: BoxDecoration(
                    color: AppColors.iconColor,
                    borderRadius: BorderRadius.circular(10),
                  ),

                ),
              ],
            )
          ),
        )
          
          ,
          FoodPageBody(),
          ],
        ),
      
    );
  }
}




