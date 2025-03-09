import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class Product {
  final String id;
  final String name;
  final double price;
  final double rating;
  final int reviewCount;
  final String imageUrl;
  final String category;
  final bool isTrending;
  final bool isOnSale;
  final double discountPercentage;
  final String description;
  final List<String> colors;
  final List<String> sizes;
  final Map<String, int> stock;

  Product({
    required this.id,
    required this.name,
    required this.price,
    required this.rating,
    required this.reviewCount,
    required this.imageUrl,
    required this.category,
    this.isTrending = false,
    this.isOnSale = false,
    this.discountPercentage = 0,
    this.description = '',
    this.colors = const [],
    this.sizes = const [],
    this.stock = const {},
  });
}

class ProductDetailPage extends StatefulWidget {
  final Product product;

  const ProductDetailPage({Key? key, required this.product}) : super(key: key);

  @override
  State<ProductDetailPage> createState() => _ProductDetailPageState();
}

class _ProductDetailPageState extends State<ProductDetailPage> {
  int _quantity = 1;
  String? _selectedColor;
  String? _selectedSize;
  int _currentImageIndex = 0;
  final PageController _pageController = PageController();

  @override
  void initState() {
    super.initState();
    if (widget.product.colors.isNotEmpty) {
      _selectedColor = widget.product.colors.first;
    }
    if (widget.product.sizes.isNotEmpty) {
      _selectedSize = widget.product.sizes.first;
    }
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void _increaseQuantity() {
    setState(() {
      _quantity++;
    });
  }

  void _decreaseQuantity() {
    if (_quantity > 1) {
      setState(() {
        _quantity--;
      });
    }
  }

  void _addToCart() {
    // Implement add to cart functionality
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('${widget.product.name} added to cart'),
        duration: const Duration(seconds: 2),
      ),
    );
    // Navigate back to previous page
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final originalPrice = widget.product.isOnSale
        ? widget.product.price / (1 - widget.product.discountPercentage / 100)
        : widget.product.price;

    return Scaffold(
      backgroundColor: Colors.white,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: Container(
          margin: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.9),
            shape: BoxShape.circle,
          ),
          child: IconButton(
            icon: const Icon(Icons.arrow_back),
            color: Colors.black,
            onPressed: () => Navigator.pop(context),
          ),
        ),
        actions: [
          Container(
            margin: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.9),
              shape: BoxShape.circle,
            ),
            child: IconButton(
              icon: const Icon(Icons.favorite_border),
              color: Colors.black,
              onPressed: () {
                // Add to favorites
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Added to favorites'),
                    duration: Duration(seconds: 2),
                  ),
                );
              },
            ),
          ),
          Container(
            margin: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.9),
              shape: BoxShape.circle,
            ),
            child: IconButton(
              icon: const Icon(Icons.share),
              color: Colors.black,
              onPressed: () {
                // Share product
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Share feature coming soon!'),
                    duration: Duration(seconds: 2),
                  ),
                );
              },
            ),
          ),
        ],
        systemOverlayStyle: SystemUiOverlayStyle.dark,
      ),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildProductImages(),
                  _buildProductInfo(originalPrice),
                  _buildColorSelection(),
                  if (widget.product.sizes.isNotEmpty) _buildSizeSelection(),
                  _buildQuantitySection(),
                  _buildDescription(),
                  _buildReviewsSection(),
                  const SizedBox(height: 100), // Space for the bottom bar
                ],
              ),
            ),
          ),
        ],
      ),
      bottomSheet: _buildBottomBar(),
    );
  }

  Widget _buildProductImages() {
    // Create a list of placeholder images if the real image URL isn't valid
    // This is where the error was happening
    List<String> images = [
      widget.product.imageUrl,
      widget.product.imageUrl,
      widget.product.imageUrl
    ];

    return Stack(
      children: [
        Container(
          height: 350,
          color: Colors.grey[100],
          child: PageView.builder(
            controller: _pageController,
            itemCount: images.length,
            onPageChanged: (index) {
              setState(() {
                _currentImageIndex = index;
              });
            },
            itemBuilder: (context, index) {
              // Use a safer way to load images with proper error handling
              return _buildImageWithErrorHandling(images[index]);
            },
          ),
        ),
        Positioned(
          bottom: 20,
          left: 0,
          right: 0,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(
              images.length,
                  (index) => Container(
                margin: const EdgeInsets.symmetric(horizontal: 4),
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _currentImageIndex == index
                      ? const Color(0xFF4C86F9)
                      : Colors.grey[300],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  // Added this method to safely handle image loading with error handling
  Widget _buildImageWithErrorHandling(String imageUrl) {
    // Check if the URL is valid before trying to load it
    if (imageUrl.startsWith('http://') || imageUrl.startsWith('https://')) {
      return Image.network(
        imageUrl,
        fit: BoxFit.contain,
        errorBuilder: (context, error, stackTrace) {
          // Return a placeholder image in case of error
          return Container(
            color: Colors.grey[200],
            child: const Center(
              child: Icon(
                Icons.image_not_supported,
                size: 50,
                color: Colors.grey,
              ),
            ),
          );
        },
        loadingBuilder: (context, child, loadingProgress) {
          if (loadingProgress == null) return child;
          return Center(
            child: CircularProgressIndicator(
              value: loadingProgress.expectedTotalBytes != null
                  ? loadingProgress.cumulativeBytesLoaded /
                  loadingProgress.expectedTotalBytes!
                  : null,
            ),
          );
        },
      );
    } else {
      // If the URL is not valid, show a placeholder
      return Container(
        color: Colors.grey[200],
        child: const Center(
          child: Icon(
            Icons.image,
            size: 50,
            color: Colors.grey,
          ),
        ),
      );
    }
  }

  Widget _buildProductInfo(double originalPrice) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.blue[50],
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              widget.product.category,
              style: TextStyle(
                color: const Color(0xFF4C86F9),
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            widget.product.name,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Icon(
                Icons.star,
                size: 18,
                color: Colors.amber,
              ),
              const SizedBox(width: 4),
              Text(
                widget.product.rating.toString(),
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 4),
              Text(
                '(${widget.product.reviewCount} reviews)',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                ),
              ),
              const Spacer(),
              if (widget.product.isOnSale)
                Text(
                  '\$${originalPrice.toStringAsFixed(2)}',
                  style: TextStyle(
                    fontSize: 16,
                    decoration: TextDecoration.lineThrough,
                    color: Colors.grey[600],
                  ),
                ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                '\$${widget.product.price.toStringAsFixed(2)}',
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF4C86F9),
                ),
              ),
              const SizedBox(width: 8),
              if (widget.product.isOnSale)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.red,
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    '${widget.product.discountPercentage.toInt()}% off',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildColorSelection() {
    if (widget.product.colors.isEmpty) return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Colors',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 12,
            children: widget.product.colors.map((color) {
              final bool isSelected = color == _selectedColor;
              // Convert string color names to actual colors
              Color actualColor;
              switch (color.toLowerCase()) {
                case 'red':
                  actualColor = Colors.red;
                  break;
                case 'blue':
                  actualColor = Colors.blue;
                  break;
                case 'green':
                  actualColor = Colors.green;
                  break;
                case 'black':
                  actualColor = Colors.black;
                  break;
                case 'white':
                  actualColor = Colors.white;
                  break;
                case 'grey':
                case 'gray':
                  actualColor = Colors.grey;
                  break;
                default:
                  actualColor = Colors.grey;
              }

              return GestureDetector(
                onTap: () {
                  setState(() {
                    _selectedColor = color;
                  });
                },
                child: Container(
                  padding: EdgeInsets.all(isSelected ? 2 : 0),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: isSelected
                        ? Border.all(color: const Color(0xFF4C86F9), width: 2)
                        : null,
                  ),
                  child: Container(
                    width: 32,
                    height: 32,
                    decoration: BoxDecoration(
                      color: actualColor,
                      shape: BoxShape.circle,
                      border: actualColor == Colors.white
                          ? Border.all(color: Colors.grey[300]!, width: 1)
                          : null,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildSizeSelection() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Size',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 12,
            children: widget.product.sizes.map((size) {
              final bool isSelected = size == _selectedSize;
              return GestureDetector(
                onTap: () {
                  setState(() {
                    _selectedSize = size;
                  });
                },
                child: Container(
                  width: 48,
                  height: 48,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    color: isSelected ? const Color(0xFF4C86F9) : Colors.white,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(
                      color: isSelected
                          ? const Color(0xFF4C86F9)
                          : Colors.grey[300]!,
                    ),
                  ),
                  child: Text(
                    size,
                    style: TextStyle(
                      color: isSelected ? Colors.white : Colors.black,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildQuantitySection() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
      child: Row(
        children: [
          const Text(
            'Quantity',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(width: 16),
          Container(
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey[300]!),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.remove),
                  onPressed: _decreaseQuantity,
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
                ),
                Text(
                  _quantity.toString(),
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.add),
                  onPressed: _increaseQuantity,
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
                ),
              ],
            ),
          ),
          const Spacer(),
          Text(
            'Stock: ${widget.product.stock[_selectedSize] ?? 'N/A'}',
            style: TextStyle(
              color: Colors.grey[600],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDescription() {
    // Default description if none provided
    String description = widget.product.description.isNotEmpty
        ? widget.product.description
        : 'This ${widget.product.name} is a high-quality product in the ${widget.product.category} category. It features premium materials and exceptional craftsmanship to provide you with an outstanding experience.';

    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Description',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            description,
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey[600],
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildReviewsSection() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Reviews',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              TextButton(
                onPressed: () {
                  // Navigate to all reviews
                },
                style: TextButton.styleFrom(
                  foregroundColor: const Color(0xFF4C86F9),
                  textStyle: const TextStyle(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                child: const Text('See All'),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _buildReviewItem(
            'Sarah Johnson',
            4.5,
            'The quality is excellent! I love the color and fit. Shipping was fast too.',
            '2 days ago',
          ),
          const SizedBox(height: 12),
          _buildReviewItem(
            'John Smith',
            5.0,
            'Absolutely perfect! Exactly what I was looking for and arrived earlier than expected.',
            '1 week ago',
          ),
        ],
      ),
    );
  }

  Widget _buildReviewItem(String name, double rating, String comment, String time) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              CircleAvatar(
                backgroundColor: Colors.grey[300],
                radius: 18,
                child: Text(
                  name.substring(0, 1),
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Row(
                    children: [
                      ...List.generate(
                        5,
                            (index) => Icon(
                          index < rating.floor()
                              ? Icons.star
                              : (index < rating ? Icons.star_half : Icons.star_border),
                          size: 16,
                          color: Colors.amber,
                        ),
                      ),
                      const SizedBox(width: 4),
                      Text(
                        rating.toString(),
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              const Spacer(),
              Text(
                time,
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            comment,
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey[800],
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomBar() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, -5),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            decoration: BoxDecoration(
              border: Border.all(color: const Color(0xFF4C86F9)),
              borderRadius: BorderRadius.circular(12),
            ),
            child: IconButton(
              icon: const Icon(
                Icons.chat_bubble_outline,
                color: Color(0xFF4C86F9),
              ),
              onPressed: () {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Chat feature coming soon!'),
                    duration: Duration(seconds: 2),
                  ),
                );
              },
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: ElevatedButton(
              onPressed: _addToCart,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF4C86F9),
                foregroundColor: Colors.white,
                minimumSize: const Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: const Text(
                'Add to Cart',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}