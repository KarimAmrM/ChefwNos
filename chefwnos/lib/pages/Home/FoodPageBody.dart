import 'package:chefwnos/utils/colors.dart';
import 'package:chefwnos/widgets/BigText.dart';
import 'package:chefwnos/widgets/SmallText.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'InventoryPage.dart';
import 'Chat.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

class FoodPageBody extends StatefulWidget {
  FoodPageBody({Key? key}) : super(key: key);

  final List<String> names = ["Chat", "Inventory"];
  @override
  _FoodPageBodyState createState() => _FoodPageBodyState();
}

class _FoodPageBodyState extends State<FoodPageBody> {
  List<FoodData> foodDataList = [];

  @override
  void initState() {
    super.initState();
    fetchDataFromApi();
  }

  void fetchDataFromApi() async {
    final response = await http.get(Uri.parse('http://localhost:105/recipes/'));
    if (response.statusCode == 200) {
      final List<dynamic> responseData = json.decode(response.body);
      setState(() {
        foodDataList =
            responseData.map((data) => FoodData.fromJson(data)).toList();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          height: 300,
          color: Colors.blueGrey,
          child: PageView.builder(
            controller: PageController(viewportFraction: 0.85),
            itemCount: foodDataList.length,
            itemBuilder: (context, position) {
              final foodData = foodDataList[position];
              return _buildPageItem(foodData);
            },
          ),
        ),
        Container(
          height: 400,
          child: ListView.builder(
            itemCount: 2,
            itemBuilder: (context, index) {
              return GestureDetector(
                onTap: () {
                  _navigateToPage(index);
                },
                child: Container(
                  margin: EdgeInsets.only(left: 20, right: 20, top: 20),
                  child: Row(
                    children: [
                      Container(
                        width: 120,
                        height: 120,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(20),
                          image: DecorationImage(
                            image: AssetImage("Assets/Images/download.png"),
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),
                      Expanded(
                        child: Container(
                          height: 110,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(20),
                            color: Colors.white,
                          ),
                          child: Padding(
                            padding: const EdgeInsets.only(left: 20, right: 20),
                            child: Column(
                              children: [
                                Padding(padding: EdgeInsets.only(top: 30)),
                                BigText(text: widget.names[index]),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildPageItem(FoodData foodData) {
    return Stack(
      children: [
        Container(
          height: 220,
          margin: EdgeInsets.only(left: 20, right: 20, top: 20),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(30),
            color: AppColors.mainColor,
            image: DecorationImage(
              image: AssetImage(foodData.imagePath),
              fit: BoxFit.cover,
            ),
          ),
        ),
        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
            height: 120,
            margin: EdgeInsets.only(left: 50, right: 50, top: 0),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(30),
              color: AppColors.mainColor,
            ),
            child: Container(
              padding: EdgeInsets.only(left: 10, right: 10, top: 20),
              child: Column(
                children: [
                  BigText(text: foodData.name),
                  SmallText(text: foodData.category),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  void _navigateToPage(int index) {
    switch (index) {
      case 0:
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => ChatPage()),
        );
        break;
      case 1:
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => InventoryPage(
              inventoryItems: [
                InventoryItem(name: 'Ingredient 1', quantity: 10),
                InventoryItem(name: 'Ingredient 2', quantity: 5),
                InventoryItem(name: 'Ingredient 3', quantity: 8),
              ],
            ),
          ),
        );
        break;
      default:
        // Handle other cases if necessary
        break;
    }
  }
}

class FoodData {
  final String name;
  final String category;
  final String imagePath;

  FoodData(
      {required this.name, required this.category, required this.imagePath});

  factory FoodData.fromJson(Map<String, dynamic> json) {
    return FoodData(
        name: json['name'],
        category: json['cuisine'],
        imagePath: "Assets/Images/download.png");
  }
}
