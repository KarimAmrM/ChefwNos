import 'package:chefwnos/utils/colors.dart';
import 'package:flutter/material.dart';

class InventoryPage extends StatefulWidget {
  final List<InventoryItem> inventoryItems;

  InventoryPage({required this.inventoryItems});

  @override
  _InventoryPageState createState() => _InventoryPageState();
}

class _InventoryPageState extends State<InventoryPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppColors.mainColor,
        title: Text('Inventory Page'),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            DataTable(
              columnSpacing: 16,
              horizontalMargin: 8,
              columns: [
                DataColumn(
                  label: Text(
                    'Ingredient Name',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                ),
                DataColumn(
                  label: Text(
                    'Quantity',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                ),
              ],
              rows: widget.inventoryItems.map((item) {
                return DataRow(cells: [
                  DataCell(Text(item.name)),
                  DataCell(Text(item.quantity.toString())),
                ]);
              }).toList(),
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: _addIngredient,
              style: ElevatedButton.styleFrom(
                primary: AppColors.mainColor,
              ),
              child: Text('Add Ingredient'),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _addIngredient,
        backgroundColor: AppColors.mainColor,
        child: Icon(Icons.add),
      ),
    );
  }

  void _addIngredient() {
    // TODO: Implement the logic to add an ingredient through an API
    setState(() {
      widget.inventoryItems.add(
        InventoryItem(name: 'New Ingredient', quantity: 0),
      );
    });
  }
}

class InventoryItem {
  final String name;
  final int quantity;

  InventoryItem({required this.name, required this.quantity});
}
