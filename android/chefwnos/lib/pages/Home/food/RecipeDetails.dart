import 'package:chefwnos/utils/AppIcon.dart';
import 'package:chefwnos/widgets/BigText.dart';
import 'package:chefwnos/widgets/ExpandalbleText.dart';
import 'package:chefwnos/widgets/smallTextVisible.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class RecipeDetails extends StatelessWidget {
const RecipeDetails({ Key? key }) : super(key: key);

  @override
  Widget build(BuildContext context){
    return Scaffold(
      body: Stack(
        children: [
          Positioned(
            left: 0,
            right: 0,
            child: 
          Container(
            height: 300,
            width: MediaQuery.of(context).size.width,
            //color: Colors.red,
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage("Assets/Images/download.png"),
                fit: BoxFit.cover,
              ),
            ),
          ),
          ),
          Positioned(
            top: 70,
            left: 20,
            right: 20,
            child:
            Row(
            children: [
              AppIcon(icon: Icons.arrow_back_ios,),

            ],
          ) 
          )
          ,Positioned(
            left: 0,
            right: 0,
            top: 280,
            bottom: 0,
            
            child: Container(
              //height: 300,
              padding: EdgeInsets.only(left: 20, right: 20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.only(topLeft: Radius.circular(20), topRight: Radius.circular(20))
              ), 
               child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    BigText(text: "3enwan",),
                    SizedBox(height: 20,),
                    BigText(text: "recipe details",),
                    SizedBox(height: 20,),
                    Expanded(child: SingleChildScrollView(child: 
                    SmallTextVisible(text: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                    ))
                    
                    ,//ExpandalbleText(text: "aaaaaaa")
                  
                  ],
                  
               ),

          ))
 
 
 
        //   ,Container(
        //     margin: EdgeInsets.only(top: 280),
        //     //TAKE ALL THE WIDTH of screen
        //     width: MediaQuery.of(context).size.width,
        //     // left: 0,
        //     // right: 0,
        //     height: 50,
        //     // top: 200,
        //     //child: Container(
        //       padding: EdgeInsets.only(left: 20, right: 20),
        //       decoration: BoxDecoration(
        //         color: Colors.white,
        //         borderRadius: BorderRadius.only(topLeft: Radius.circular(20), topRight: Radius.circular(20)),
        //       ),
        //       child: Column(
        //         children: [
        //           BigText(text: "3enwan",size: 40,)
        //         ],
        //       ),
              
               
              
        //   )
        //  // )
        //  ,Container(
        //   margin: EdgeInsets.only(top: 350),
        //   color: Colors.greenAccent,
        //   child: Column(
        //     crossAxisAlignment: CrossAxisAlignment.start,
        //   children: [
        //     BigText(text: "Recipe Details"),
        //     Expanded(child:  SingleChildScrollView(
        //     child:ExpandalbleText(text: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        //     ))
           
        //   ],
        //  )

        //  )
        ],
      ),
    );
  }
}