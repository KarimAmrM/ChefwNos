import 'package:flutter/material.dart';

class NavItem {
  
    int id;
    String icon;
    Widget destination;

NavItem({required this.id,required this.icon,required this.destination});

// If there is no destination then it help us
  bool destinationChecker() {
    if (destination != null) {
      return true;
    }
    return false;
  }
}
