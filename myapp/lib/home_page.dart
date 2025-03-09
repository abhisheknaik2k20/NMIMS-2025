import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:myapp/AUTH_LOGIN/SCREEN/login_screen.dart';
import 'package:myapp/main.dart';
import 'card_page.dart';
import 'cart_page.dart';
import 'card_page.dart' as card_page; // Import with namespace

class Product {
  final String id;
  final String name;
  final double price;
  final double rating;
  final int reviewCount;
  final IconData icon; // Using icons instead of image URLs
  final String category;
  final bool isTrending;
  final bool isOnSale;
  final double discountPercentage;
  final String description;
  final List<dynamic> colors;
  final List<dynamic> sizes;
  final Map<dynamic, dynamic> stock;

  Product({
    required this.id,
    required this.name,
    required this.price,
    required this.rating,
    required this.reviewCount,
    required this.icon,
    required this.category,
    this.isTrending = false,
    this.isOnSale = false,
    this.discountPercentage = 0,
    this.description = '',
    this.colors = const [],
    this.sizes = const [],
    this.stock = const {},
  });

// Fix for the toCardPageProduct method in home_page.dart
  card_page.Product toCardPageProduct() {
    // Convert the nested map structure correctly
    Map<String, int> flattenedStock = {};

    // Flatten the nested map structure if it's in the format you're using
    // Your current stock structure is: {'Blue': {'S': 5, 'M': 10}}
    if (stock is Map) {
      try {
        // For each color
        stock.forEach((color, sizeMap) {
          if (color is String && sizeMap is Map) {
            // For each size in that color
            sizeMap.forEach((size, quantity) {
              if (size is String && quantity is int) {
                // Create a key like "Blue-S" to represent color-size combination
                String key = "$color-$size";
                flattenedStock[key] = quantity;
              }
            });
          }
        });
      } catch (e) {
        print("Error flattening stock: $e");
        flattenedStock = {};
      }
    }

    return card_page.Product(
      id: id,
      name: name,
      price: price,
      rating: rating,
      reviewCount: reviewCount,
      imageUrl: icon.toString(),
      // Convert IconData to a string for imageUrl
      category: category,
      isTrending: isTrending,
      isOnSale: isOnSale,
      discountPercentage: discountPercentage,
      description: description,
      colors: colors.map((color) => color.toString()).toList(),
      // Convert all elements to String
      sizes: sizes.map((size) => size.toString()).toList(),
      // Convert all elements to String
      stock: flattenedStock, // Use the flattened stock map
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;
  String _searchQuery = '';
  int _currentPromotionIndex = 0;

  final List<Product> products = [
    // Fashion Category
    Product(
      id: 'f1',
      name: 'Premium Denim Jacket',
      price: 89.99,
      rating: 4.9,
      reviewCount: 345,
      icon: Icons.checkroom,
      category: 'Fashion',
      isTrending: true,
      isOnSale: true,
      discountPercentage: 10,
      description:
          'Classic denim jacket with premium quality stitching and comfortable fit.',
      colors: ['Blue', 'Black', 'Grey'],
      sizes: ['S', 'M', 'L', 'XL'],
      stock: {
        'Blue': {'S': 5, 'M': 10, 'L': 8, 'XL': 6},
        'Black': {'S': 7, 'M': 9, 'L': 12, 'XL': 4},
        'Grey': {'S': 3, 'M': 7, 'L': 5, 'XL': 2}
      },
    ),
    Product(
      id: 'f2',
      name: 'Slim Fit Chinos',
      price: 49.99,
      rating: 4.7,
      reviewCount: 215,
      icon: Icons.checkroom,
      category: 'Fashion',
      isTrending: false,
      description:
          'Modern slim fit chinos perfect for casual and semi-formal occasions.',
      colors: ['Khaki', 'Navy', 'Black'],
      sizes: ['30', '32', '34', '36'],
      stock: {
        'Khaki': {'30': 8, '32': 12, '34': 10, '36': 5},
        'Navy': {'30': 6, '32': 8, '34': 7, '36': 4},
        'Black': {'30': 9, '32': 11, '34': 8, '36': 6}
      },
    ),
    Product(
      id: 'f3',
      name: 'Cotton Summer Dress',
      price: 59.99,
      rating: 4.8,
      reviewCount: 178,
      icon: Icons.checkroom,
      category: 'Fashion',
      isOnSale: true,
      discountPercentage: 15,
      description: 'Light and breezy cotton dress, perfect for summer days.',
      colors: ['White', 'Yellow', 'Pink'],
      sizes: ['XS', 'S', 'M', 'L'],
      stock: {
        'White': {'XS': 4, 'S': 7, 'M': 9, 'L': 5},
        'Yellow': {'XS': 3, 'S': 6, 'M': 8, 'L': 4},
        'Pink': {'XS': 5, 'S': 8, 'M': 10, 'L': 6}
      },
    ),
    Product(
      id: 'j7',
      name: 'Sterling Silver Pendant Necklace',
      price: 79.99,
      rating: 4.7,
      reviewCount: 142,
      icon: Icons.diamond,
      category: 'Jewelry',
      isOnSale: true,
      discountPercentage: 20,
      description:
          'Elegant sterling silver pendant with delicate chain, perfect for everyday wear or special occasions.',
      colors: ['Silver', 'Rose Gold', 'Gold'],
      sizes: ['16"', '18"', '20"', '24"'],
      stock: {
        'Silver': {'16"': 8, '18"': 12, '20"': 6, '24"': 4},
        'Rose Gold': {'16"': 6, '18"': 9, '20"': 5, '24"': 3},
        'Gold': {'16"': 5, '18"': 7, '20"': 4, '24"': 2}
      },
    ),
    Product(
      id: 'f4',
      name: 'Leather Crossbody Bag',
      price: 79.99,
      rating: 4.6,
      reviewCount: 142,
      icon: Icons.shopping_bag,
      category: 'Fashion',
      description:
          'Genuine leather crossbody bag with adjustable strap and multiple compartments.',
      colors: ['Brown', 'Black', 'Tan'],
      sizes: ['One Size'],
      stock: {
        'Brown': {'One Size': 15},
        'Black': {'One Size': 18},
        'Tan': {'One Size': 12}
      },
    ),

    // Electronics Category
    Product(
      id: 'e1',
      name: 'Smart Wireless Earbuds',
      price: 129.99,
      rating: 4.9,
      reviewCount: 345,
      icon: Icons.headphones,
      category: 'Electronics',
      isTrending: true,
      description:
          'True wireless earbuds with noise cancellation and long battery life.',
      colors: ['White', 'Black'],
      sizes: ['One Size'],
      stock: {
        'White': {'One Size': 25},
        'Black': {'One Size': 30}
      },
    ),
    Product(
      id: 'e2',
      name: 'Bluetooth Speaker',
      price: 79.99,
      rating: 4.8,
      reviewCount: 230,
      icon: Icons.speaker,
      category: 'Electronics',
      isTrending: true,
      isOnSale: true,
      discountPercentage: 20,
      description:
          'Portable Bluetooth speaker with 360° sound and waterproof design.',
      colors: ['Black', 'Blue', 'Red'],
      sizes: ['One Size'],
      stock: {
        'Black': {'One Size': 20},
        'Blue': {'One Size': 15},
        'Red': {'One Size': 18}
      },
    ),
    Product(
      id: 'e3',
      name: 'Smartwatch Series 5',
      price: 199.99,
      rating: 4.9,
      reviewCount: 412,
      icon: Icons.watch,
      category: 'Electronics',
      isTrending: true,
      description:
          'Advanced smartwatch with health monitoring and customizable watch faces.',
      colors: ['Silver', 'Black', 'Rose Gold'],
      sizes: ['40mm', '44mm'],
      stock: {
        'Silver': {'40mm': 12, '44mm': 15},
        'Black': {'40mm': 14, '44mm': 18},
        'Rose Gold': {'40mm': 10, '44mm': 8}
      },
    ),
    Product(
      id: 'e4',
      name: 'Noise Cancelling Headphones',
      price: 249.99,
      rating: 4.7,
      reviewCount: 187,
      icon: Icons.headphones,
      category: 'Electronics',
      description:
          'Over-ear headphones with active noise cancellation and premium audio quality.',
      colors: ['Black', 'Silver'],
      sizes: ['One Size'],
      stock: {
        'Black': {'One Size': 22},
        'Silver': {'One Size': 18}
      },
    ),

    // Books Category
    Product(
      id: 'b1',
      name: 'The Silent Patient',
      price: 19.99,
      rating: 4.8,
      reviewCount: 276,
      icon: Icons.book,
      category: 'Book',
      description:
          'A psychological thriller about a woman\'s act of violence against her husband.',
      colors: ['Hardcover', 'Paperback', 'eBook'],
      sizes: ['One Size'],
      stock: {
        'Hardcover': {'One Size': 35},
        'Paperback': {'One Size': 50},
        'eBook': {'One Size': 999}
      },
    ),
    Product(
      id: 'b2',
      name: 'Atomic Habits',
      price: 24.99,
      rating: 4.9,
      reviewCount: 345,
      icon: Icons.book,
      category: 'Book',
      isTrending: true,
      description:
          'An easy and proven way to build good habits and break bad ones.',
      colors: ['Hardcover', 'Paperback', 'eBook'],
      sizes: ['One Size'],
      stock: {
        'Hardcover': {'One Size': 40},
        'Paperback': {'One Size': 60},
        'eBook': {'One Size': 999}
      },
    ),
    Product(
      id: 'b3',
      name: 'Where the Crawdads Sing',
      price: 22.99,
      rating: 4.7,
      reviewCount: 214,
      icon: Icons.book,
      category: 'Book',
      description:
          'A novel about a young woman who grows up in isolation in the marshes of North Carolina.',
      colors: ['Hardcover', 'Paperback', 'eBook'],
      sizes: ['One Size'],
      stock: {
        'Hardcover': {'One Size': 30},
        'Paperback': {'One Size': 45},
        'eBook': {'One Size': 999}
      },
    ),
    Product(
      id: 'b4',
      name: 'Psychology of Money',
      price: 27.99,
      rating: 4.8,
      reviewCount: 189,
      icon: Icons.book,
      category: 'Book',
      description: 'Timeless lessons on wealth, greed, and happiness.',
      colors: ['Hardcover', 'Paperback', 'eBook'],
      sizes: ['One Size'],
      stock: {
        'Hardcover': {'One Size': 25},
        'Paperback': {'One Size': 38},
        'eBook': {'One Size': 999}
      },
    ),

    // Sports Category
    Product(
      id: 's1',
      name: 'Running Shoes',
      price: 99.99,
      rating: 4.9,
      reviewCount: 325,
      icon: Icons.directions_run,
      category: 'Sport',
      isTrending: true,
      isOnSale: true,
      discountPercentage: 30,
      description:
          'Lightweight and responsive running shoes with advanced cushioning.',
      colors: ['Black/White', 'Blue/Grey', 'Red/Black'],
      sizes: ['7', '8', '9', '10', '11', '12'],
      stock: {
        'Black/White': {'7': 5, '8': 8, '9': 10, '10': 12, '11': 8, '12': 6},
        'Blue/Grey': {'7': 4, '8': 6, '9': 9, '10': 10, '11': 7, '12': 5},
        'Red/Black': {'7': 3, '8': 7, '9': 8, '10': 11, '11': 6, '12': 4}
      },
    ),
    Product(
      id: 's2',
      name: 'Yoga Mat Premium',
      price: 49.99,
      rating: 4.7,
      reviewCount: 198,
      icon: Icons.fitness_center,
      category: 'Sport',
      isOnSale: true,
      discountPercentage: 25,
      description:
          'Non-slip yoga mat with extra cushioning for joint protection.',
      colors: ['Purple', 'Blue', 'Green'],
      sizes: ['Regular', 'Extra Long'],
      stock: {
        'Purple': {'Regular': 18, 'Extra Long': 12},
        'Blue': {'Regular': 15, 'Extra Long': 10},
        'Green': {'Regular': 20, 'Extra Long': 14}
      },
    ),
    Product(
      id: 's3',
      name: 'Fitness Tracker',
      price: 89.99,
      rating: 4.8,
      reviewCount: 245,
      icon: Icons.watch,
      category: 'Sport',
      isTrending: true,
      description:
          'Tracks steps, heart rate, sleep, and multiple workout types.',
      colors: ['Black', 'Blue', 'Pink'],
      sizes: ['One Size'],
      stock: {
        'Black': {'One Size': 30},
        'Blue': {'One Size': 25},
        'Pink': {'One Size': 20}
      },
    ),
    Product(
      id: 's4',
      name: 'Basketball Indoor/Outdoor',
      price: 39.99,
      rating: 4.6,
      reviewCount: 167,
      icon: Icons.sports_basketball,
      category: 'Sport',
      description:
          'Official size basketball suitable for both indoor and outdoor play.',
      colors: ['Orange', 'Brown'],
      sizes: ['Size 7', 'Size 6', 'Size 5'],
      stock: {
        'Orange': {'Size 7': 15, 'Size 6': 12, 'Size 5': 10},
        'Brown': {'Size 7': 14, 'Size 6': 10, 'Size 5': 8}
      },
    ),

    // Skincare Category
    Product(
      id: 'sk1',
      name: 'Hydrating Face Serum',
      price: 59.99,
      rating: 4.9,
      reviewCount: 312,
      icon: Icons.spa,
      category: 'Skincare',
      isTrending: true,
      description: 'Hydrating serum with hyaluronic acid for all skin types.',
      colors: ['30ml', '50ml'],
      sizes: ['One Size'],
      stock: {
        '30ml': {'One Size': 25},
        '50ml': {'One Size': 20}
      },
    ),
    Product(
      id: 'sk2',
      name: 'Charcoal Face Mask',
      price: 29.99,
      rating: 4.7,
      reviewCount: 186,
      icon: Icons.spa,
      category: 'Skincare',
      description:
          'Deep cleansing charcoal mask that removes impurities and excess oil.',
      colors: ['100ml', '200ml'],
      sizes: ['One Size'],
      stock: {
        '100ml': {'One Size': 30},
        '200ml': {'One Size': 25}
      },
    ),
    Product(
      id: 'sk3',
      name: 'Anti-Aging Night Cream',
      price: 69.99,
      rating: 4.8,
      reviewCount: 234,
      icon: Icons.spa,
      category: 'Skincare',
      isOnSale: true,
      discountPercentage: 15,
      description:
          'Nourishing night cream with retinol and peptides to reduce fine lines.',
      colors: ['50ml', '75ml'],
      sizes: ['One Size'],
      stock: {
        '50ml': {'One Size': 22},
        '75ml': {'One Size': 18}
      },
    ),
    Product(
      id: 'sk4',
      name: 'Vitamin C Brightening Set',
      price: 89.99,
      rating: 4.9,
      reviewCount: 264,
      icon: Icons.spa,
      category: 'Skincare',
      description:
          'Complete set with vitamin C serum, moisturizer, and eye cream for brighter skin.',
      colors: ['Standard', 'Deluxe'],
      sizes: ['One Size'],
      stock: {
        'Standard': {'One Size': 15},
        'Deluxe': {'One Size': 12}
      },
    ),
  ];

  List<Product> get filteredProducts {
    if (_searchQuery.isEmpty) {
      return products;
    }
    return products
        .where((product) =>
            product.name.toLowerCase().contains(_searchQuery.toLowerCase()) ||
            product.category.toLowerCase().contains(_searchQuery.toLowerCase()))
        .toList();
  }

  List<Product> get trendingProducts {
    return products.where((product) => product.isTrending).toList();
  }

  List<String> get categories {
    return products.map((product) => product.category).toSet().toList();
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });

    if (index == 3) {
      // Chat tab
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Chat feature coming soon!'),
          duration: Duration(seconds: 2),
        ),
      );
    } else if (index == 2) {
      // Favorites tab
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Favorites feature coming soon!'),
          duration: Duration(seconds: 2),
        ),
      );
    } else if (index == 4) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Profile feature coming soon!'),
          duration: Duration(seconds: 2),
        ),
      );
    } else if (index == 1) {
      // Cart tab
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => CartPage()),
      );
    }
  }

  void _navigateToProductDetail(Product product) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ProductDetailPage(
          product: product.toCardPageProduct(),
        ),
      ),
    );
  }

  Future<void> showBreachNotification() async {
    const AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
      'security_alerts', // Channel ID
      'Security Alerts', // Channel Name
      channelDescription: 'Notifies users about security breaches',
      importance: Importance.max,
      priority: Priority.high,
      ticker: 'Security Alert',
      playSound: true,
    );

    const NotificationDetails notificationDetails =
        NotificationDetails(android: androidDetails);

    await flutterLocalNotificationsPlugin.show(
      0, // Notification ID
      'Security Alert: Potential Data Breach 🚨',
      '''
We detected a potentially unauthorized login attempt.

Details:
- Time: ${DateTime.now().toIso8601String()}
- IP Address: 192.168.1.102

If this was not you, please reset your password immediately.
''',
      notificationDetails,
    );
  }

  void callfunc() async {
    showBreachNotification();
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<DocumentSnapshot>(
      stream: FirebaseFirestore.instance
          .collection('INTRUDER')
          .doc('intruder')
          .snapshots(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasData && snapshot.data != null) {
          if (snapshot.data!['iam']) callfunc();
        }

        // Now use the boolean value to control your UI
        return Scaffold(
          body: SafeArea(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHeader(),
                  _buildSearchBar(),
                  _buildCategorySection(),
                  _buildTrendingProducts(),
                  _buildAllProducts(),
                  const SizedBox(height: 16),
                ],
              ),
            ),
          ),
          bottomNavigationBar: _buildBottomNavigationBar(),
        );
      },
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Row(
        children: [
          GestureDetector(
            onTap: () {
              FirebaseAuth.instance.signOut();
              Navigator.of(context).pushAndRemoveUntil(
                MaterialPageRoute(builder: (context) => LoginSignupScreen()),
                (Route route) => false,
              );
            },
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                shape: BoxShape.circle,
              ),
              child: const Center(
                child: Text(
                  'AN',
                  style: TextStyle(
                    color: Colors.black87,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Abhishek Naik',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Row(
                  children: [
                    Icon(
                      Icons.location_on,
                      size: 14,
                      color: Colors.grey[600],
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'Mumbai, India',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                    Icon(
                      Icons.keyboard_arrow_down,
                      size: 14,
                      color: Colors.grey[600],
                    ),
                  ],
                ),
              ],
            ),
          ),
          Container(
            decoration: BoxDecoration(
              color: Colors.grey[100],
              borderRadius: BorderRadius.circular(8),
            ),
            child: Stack(
              alignment: Alignment.topRight,
              children: [
                IconButton(
                  icon: const Icon(Icons.notifications_none),
                  onPressed: () {},
                ),
                Positioned(
                  right: 8,
                  top: 8,
                  child: Container(
                    padding: const EdgeInsets.all(4),
                    decoration: const BoxDecoration(
                      color: Colors.red,
                      shape: BoxShape.circle,
                    ),
                    constraints: const BoxConstraints(
                      minWidth: 8,
                      minHeight: 8,
                    ),
                    child: const Text(
                      '2',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSearchBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.grey[100],
          borderRadius: BorderRadius.circular(12),
        ),
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Row(
          children: [
            Icon(
              Icons.search,
              color: Colors.grey[600],
            ),
            const SizedBox(width: 8),
            Expanded(
              child: TextField(
                decoration: const InputDecoration(
                  hintText: 'What are you looking for',
                  border: InputBorder.none,
                  hintStyle: TextStyle(fontSize: 14),
                ),
                onChanged: (value) {
                  setState(() {
                    _searchQuery = value;
                  });
                },
              ),
            ),
            IconButton(
              icon: const Icon(Icons.qr_code_scanner),
              color: Colors.grey[600],
              onPressed: () {},
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCategorySection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Shop by category',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                'See all',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.blue[600],
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
        SizedBox(
          height: 100,
          child: ListView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            children: [
              _buildCategoryItem('Fashion', Icons.checkroom),
              _buildCategoryItem('Electronics', Icons.devices),
              _buildCategoryItem('Book', Icons.book),
              _buildCategoryItem('Sport', Icons.sports_soccer),
              _buildCategoryItem('Skincare', Icons.spa),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildCategoryItem(String name, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(right: 16),
      child: Column(
        children: [
          Container(
            width: 60,
            height: 60,
            decoration: BoxDecoration(
              color: Colors.grey[200],
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: Colors.black87),
          ),
          const SizedBox(height: 8),
          Text(
            name,
            style: const TextStyle(fontSize: 12),
          ),
        ],
      ),
    );
  }

  Widget _buildPromotionCarousel() {
    return Container(
      height: 150, // Reduced height to prevent overflow
      margin: const EdgeInsets.symmetric(vertical: 16),
      child: PageView(
        onPageChanged: (index) {
          setState(() {
            _currentPromotionIndex = index;
          });
        },
        children: [
          _buildPromotionCard(
            'Up to 90% off on sports products',
            'Celebrating National Sports Day',
            const Color(0xFF1E1E1E),
            Icons.sports_soccer,
          ),
          _buildPromotionCard(
            'Flash Sale on Electronics',
            'Limited time offers on premium gadgets',
            const Color(0xFF0D47A1),
            Icons.devices,
          ),
          _buildPromotionCard(
            'New Arrivals in Fashion',
            'Upgrade your wardrobe today',
            const Color(0xFF212121),
            Icons.checkroom,
          ),
        ],
      ),
    );
  }

  Widget _buildPromotionCard(
      String title, String subtitle, Color color, IconData icon) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Expanded(
            flex: 3,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 16, // Reduced font size
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 0),
                Text(
                  subtitle,
                  style: TextStyle(
                    fontSize: 12, // Reduced font size
                    color: Colors.white.withOpacity(0.8),
                  ),
                ),
                const SizedBox(height: 7), // Reduced spacing
                ElevatedButton(
                  onPressed: () {},
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: color,
                    minimumSize: const Size(100, 20),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  child: const Text('Shop Now'),
                ),
              ],
            ),
          ),
          Expanded(
            flex: 2,
            child: Center(
              child: Icon(
                icon,
                size: 60,
                color: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrendingProducts() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Trending product',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                'View more',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.blue[600],
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
        SizedBox(
          height: 240,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            itemCount: trendingProducts.length,
            itemBuilder: (context, index) {
              return GestureDetector(
                onTap: () => _navigateToProductDetail(trendingProducts[index]),
                child: _buildProductCard(trendingProducts[index]),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildAllProducts() {
    if (_searchQuery.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text(
            'Search Results',
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        filteredProducts.isEmpty
            ? Center(
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Text(
                    'No products found for "$_searchQuery"',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey[600],
                    ),
                  ),
                ),
              )
            : GridView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                padding: const EdgeInsets.symmetric(horizontal: 16),
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2,
                  childAspectRatio: 0.75,
                  crossAxisSpacing: 16,
                  mainAxisSpacing: 16,
                ),
                itemCount: filteredProducts.length,
                itemBuilder: (context, index) {
                  return GestureDetector(
                    onTap: () =>
                        _navigateToProductDetail(filteredProducts[index]),
                    child: _buildProductCard(filteredProducts[index]),
                  );
                },
              ),
      ],
    );
  }

  Widget _buildProductCard(Product product) {
    return Container(
      width: 160,
      margin: const EdgeInsets.only(right: 16),
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
          // Product icon/image
          Stack(
            children: [
              Container(
                height: 120,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(12),
                    topRight: Radius.circular(12),
                  ),
                ),
                child: Icon(
                  product.icon,
                  size: 50,
                  color: Colors.black54,
                ),
              ),
              if (product.isOnSale)
                Positioned(
                  left: 8,
                  top: 8,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.red,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      '${product.discountPercentage.toInt()}% off',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              Positioned(
                right: 8,
                top: 8,
                child: Container(
                  padding: const EdgeInsets.all(4),
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    Icons.favorite_border,
                    size: 18,
                    color: Colors.grey[600],
                  ),
                ),
              ),
            ],
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  product.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '\$${product.price.toStringAsFixed(2)}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF4C86F9),
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Icon(
                      Icons.star,
                      size: 16,
                      color: Colors.amber,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      product.rating.toString(),
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      '(${product.reviewCount})',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
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

  Widget _buildBottomNavigationBar() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            spreadRadius: 1,
            blurRadius: 5,
            offset: const Offset(0, -1),
          ),
        ],
      ),
      child: BottomNavigationBar(
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home_outlined),
            activeIcon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.shopping_cart_outlined),
            activeIcon: Icon(Icons.shopping_cart),
            label: 'Cart',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.favorite_border),
            activeIcon: Icon(Icons.favorite),
            label: 'Favorite',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.chat_bubble_outline),
            activeIcon: Icon(Icons.chat_bubble),
            label: 'Chat',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_outline),
            activeIcon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: const Color(0xFF4C86F9),
        unselectedItemColor: Colors.grey[600],
        showUnselectedLabels: true,
        selectedLabelStyle: const TextStyle(fontSize: 12),
        unselectedLabelStyle: const TextStyle(fontSize: 12),
        type: BottomNavigationBarType.fixed,
        backgroundColor: Colors.white,
        elevation: 0,
        onTap: _onItemTapped,
      ),
    );
  }
}
