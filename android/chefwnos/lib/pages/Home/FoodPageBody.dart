import 'package:chefwnos/utils/colors.dart';
import 'package:chefwnos/widgets/BigText.dart';
import 'package:chefwnos/widgets/SmallText.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class FoodPageBody extends StatefulWidget {
  const FoodPageBody({ Key? key }) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _FoodPageBodyState createState() => _FoodPageBodyState();
}

class _FoodPageBodyState extends State<FoodPageBody> {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
      height: 300,
      color: Colors.blueGrey,
      child: PageView.builder(
        controller: PageController(viewportFraction: 0.85),
        itemCount: 5,
        //controller: PageController(initialPage: 1, keepPage: true, viewportFraction: 1),
        itemBuilder: (context, position){
        
        return _buildPageItem(position);
      }),
      
     ),
    Container(
      height: 400,
      //color: Colors.blueGrey,
      child: ListView.builder(
      itemCount: 2,
      itemBuilder: (context, index){
        return Container(
          // height: 100,
          // color: Colors.red,
          margin: EdgeInsets.only(left: 20, right: 20 , top: 20),
          child: Row(
            children: [
              Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  //color: AppColors.mainColor,
                  image: DecorationImage(
                    image: AssetImage("Assets/Images/download.png"),
                    fit: BoxFit.cover,
                  ),
                ),
              )
              ,Expanded(child: 
              Container(
                height: 110,
                //width: 220,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  color: Colors.white,
                ),
                child: Padding(
                  padding: const EdgeInsets.only(left: 20, right: 20),
                  child: Column(
                    children: [
                      Padding(padding: EdgeInsets.only(top: 30),),
                      BigText(text: "3enwan")

                    ],
                  ),
              )
              )
              )
            ],
          ),
        );
      }
    ),
    )
    ],
    );
  }
  Widget _buildPageItem(int index){
    return Stack(
      children: [
        Container(
        height: 220,
        margin: EdgeInsets.only(left: 20, right: 20 , top: 20),
        decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(30),
        color: AppColors.mainColor,
        image: DecorationImage(
          image: AssetImage("Assets/Images/food1.png"),
          fit: BoxFit.cover,
        ),
        ),
      ),

        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
          height: 120,
          //padding: EdgeInsets.only(left: 50, right: 50,bottom: 20 ),
          margin: EdgeInsets.only(left: 50, right: 50 , top: 0),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(30),
            color: AppColors.mainColor,
            ),
            child: Container(
              padding: EdgeInsets.only(left: 10, right: 10,top: 20 ),
              child: Column(
                children: [
                  BigText(text: "3enwan"),
                  SmallText(text: "shar7")
                ],
              ),
            ),
            ),
          ),
        
      ],
    );


    
  }
}