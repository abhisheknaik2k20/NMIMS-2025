import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class CartPage extends StatelessWidget {
  const CartPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'My Cart',
          style: TextStyle(
            fontWeight: FontWeight.bold,
          ),
        ),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildCartItem(
            'Apple iPhone 15 Pro 128GB Natural Titanium',
            699.99,
            Icons.smartphone,
            '1',
            context,
          ),
          const SizedBox(height: 16),
          _buildCartItem(
            'Bluetooth Speaker',
            79.99,
            Icons.speaker,
            '1',
            context,
          ),
          const SizedBox(height: 24),
          _buildPromoCodeField(),
          const SizedBox(height: 24),
          _buildOrderSummary(),
          const SizedBox(height: 24),
          _buildCheckoutButton(),
        ],
      ),
    );
  }

  Widget _buildCartItem(String name, double price, IconData iconData,
      String quantity, BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 70,
            height: 70,
            decoration: BoxDecoration(
              color: Colors.grey[100],
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(
              iconData,
              size: 30,
              color: const Color(0xFF4C86F9),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  name,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 8),
                Text(
                  '\$${price.toStringAsFixed(2)}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF4C86F9),
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey.shade300),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        children: [
                          InkWell(
                            onTap: () {},
                            child: Container(
                              padding: const EdgeInsets.all(4),
                              child: const Icon(Icons.remove, size: 14),
                            ),
                          ),
                          Text(
                            quantity,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          InkWell(
                            onTap: () {},
                            child: Container(
                              padding: const EdgeInsets.all(4),
                              child: const Icon(Icons.add, size: 14),
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      icon: const Icon(
                        Icons.delete_outline,
                        color: Colors.red,
                        size: 20,
                      ),
                      constraints: const BoxConstraints(),
                      padding: EdgeInsets.zero,
                      onPressed: () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Item removed from cart'),
                            duration: Duration(seconds: 2),
                          ),
                        );
                      },
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPromoCodeField() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          const Expanded(
            child: TextField(
              decoration: InputDecoration(
                hintText: 'Promo Code',
                border: InputBorder.none,
                hintStyle: TextStyle(fontSize: 14),
                isCollapsed: true,
                contentPadding: EdgeInsets.zero,
              ),
            ),
          ),
          TextButton(
            onPressed: () {},
            style: TextButton.styleFrom(
              foregroundColor: const Color(0xFF4C86F9),
              textStyle: const TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),
            child: const Text('Apply'),
          ),
        ],
      ),
    );
  }

  Widget _buildOrderSummary() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Order Summary',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          _buildSummaryRow('Subtotal', '\$779.98'),
          _buildSummaryRow('Shipping', '\$9.99'),
          _buildSummaryRow('Tax', '\$77.99'),
          const Divider(height: 24),
          _buildSummaryRow(
            'Total',
            '\$867.96',
            isBold: true,
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryRow(String title, String value, {bool isBold = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: isBold ? 16 : 14,
              fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
              color: isBold ? Colors.black : Colors.grey[600],
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: isBold ? 16 : 14,
              fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
              color: isBold ? const Color(0xFF4C86F9) : Colors.black,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCheckoutButton() {
    // Add a state variable to track loading state
    final ValueNotifier<bool> _isLoading = ValueNotifier<bool>(false);

    return ValueListenableBuilder<bool>(
      valueListenable: _isLoading,
      builder: (context, isLoading, child) {
        return ElevatedButton(
          onPressed: isLoading
              ? null // Disable button when loading
              : () async {
                  // Set loading to true
                  _isLoading.value = true;

                  CollectionReference transactions =
                      FirebaseFirestore.instance.collection('purchases');

                  // Sample transaction data
                  List<Map<String, dynamic>> checkoutData = [
                    {
                      "user_id": 1,
                      "amount": 1500.00,
                      "category": "Electronics",
                      "timestamp": DateTime.now().toIso8601String(),
                    },
                  ];

                  try {
                    // Add each transaction to Firestore
                    for (var data in checkoutData) {
                      await transactions.add(data);
                    }
                    print("Checkout data saved successfully!");

                    // Keep showing the loading indicator for 2 seconds after completion
                    await Future.delayed(const Duration(seconds: 2));

                    // Set loading to false
                    _isLoading.value = false;

                    // Show success alert
                    _showAlert(context, 'Success',
                        'Your order has been placed successfully.', true);
                  } catch (e) {
                    print("Error saving data: $e");

                    // Keep showing the loading indicator for 2 seconds even on error
                    await Future.delayed(const Duration(seconds: 2));

                    // Set loading to false
                    _isLoading.value = false;

                    // Show error alert
                    _showAlert(context, 'Error',
                        'Failed to place your order. Please try again.', false);
                  }
                },
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF4C86F9),
            foregroundColor: Colors.white,
            minimumSize: const Size(double.infinity, 50),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            // Add disabled style to maintain visual consistency when loading
            disabledBackgroundColor: const Color(0xFF4C86F9).withOpacity(0.7),
            disabledForegroundColor: Colors.white.withOpacity(0.7),
          ),
          child: isLoading
              ? const SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(
                    color: Colors.white,
                    strokeWidth: 3,
                  ),
                )
              : const Text(
                  'Proceed to Checkout',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
        );
      },
    );
  }

// Helper method to show an alert dialog
  void _showAlert(
      BuildContext context, String title, String message, bool isSuccess) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          title: Row(
            children: [
              Icon(
                isSuccess ? Icons.dangerous : Icons.error,
                color: Colors.red,
                size: 30,
              ),
              Text("FRAUD TRANSACTION"),
            ],
          ),
          content: Text("YOUR ORDER IS DROPPED"),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: const Text(
                'OK',
                style: TextStyle(
                  color: Color(0xFF4C86F9),
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
