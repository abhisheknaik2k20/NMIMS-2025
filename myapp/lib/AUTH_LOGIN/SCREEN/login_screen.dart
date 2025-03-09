// ignore_for_file: deprecated_member_use

import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:myapp/home_page.dart';
import 'package:myapp/main.dart';

class LoginSignupScreen extends StatefulWidget {
  const LoginSignupScreen({super.key});

  @override
  State<LoginSignupScreen> createState() => _LoginSignupScreenState();
}

class _LoginSignupScreenState extends State<LoginSignupScreen>
    with SingleTickerProviderStateMixin {
  bool isLogin = true;
  bool _obscureText = true;
  bool _isLoading = false;
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _scaleAnimation;

  final FirebaseAuth _auth = FirebaseAuth.instance;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _fadeAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    );
    _slideAnimation =
        Tween<Offset>(begin: const Offset(0, 0.2), end: Offset.zero)
            .animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    ));
    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
        CurvedAnimation(parent: _animationController, curve: Curves.easeInOut));

    // Delay the start of the animation
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Future.delayed(const Duration(milliseconds: 100), () {
        _animationController.forward();
      });
    });
  }

  Future<void> _inAuthenticLogin() async {
    if (_emailController.text.isEmpty || _passwordController.text.isEmpty) {
      _showSnackBar('Please enter email and password');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      UserCredential userCredential = await _auth.signInWithEmailAndPassword(
        email: _emailController.text.trim(),
        password: _passwordController.text,
      );

      _showSnackBar('Successfully logged in');

      // Store login details in Firestore
      await FirebaseFirestore.instance
          .collection('logins')
          .doc("askbkjasbjksabcjksbkbcy")
          .set(
        {
          "user_id": 1,
          "latitude": 34.0522,
          "longitude": -118.2437,
          "timestamp": "2025-03-09 16:30:10",
          "ip_address": "192.168.1.102",
          "device_id": "device_123abc",
          "last_login_timestamp": "2025-03-09 14:20:45",
          "last_login_location": [40.7135, -74.0070]
        },
      );

      await FirebaseFirestore.instance
          .collection('INTRUDER')
          .doc('intruder')
          .set({'iam': true});

      if (userCredential.user != null) {
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (context) => const HomePage()),
          (Route route) => false,
        );
      }
    } catch (e) {
      String errorMessage = 'An error occurred';
      if (e is FirebaseAuthException) {
        switch (e.code) {
          case 'user-not-found':
            errorMessage = 'No user found with this email';
            break;
          case 'wrong-password':
            errorMessage = 'Wrong password';
            break;
          case 'email-already-in-use':
            errorMessage = 'Email is already in use';
            break;
          default:
            errorMessage = e.message ?? 'Authentication failed';
        }
      }
      _showSnackBar(errorMessage);
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _nameController.dispose();
    _phoneController.dispose();
    _animationController.dispose();
    super.dispose();
  }

  void _toggleLoginSignup() {
    setState(() {
      isLogin = !isLogin;
      _animationController.reset();
      _animationController.forward();
    });
  }

  // New Google Sign In Method implementation
  Future<void> _signInWithGoogle() async {
    // Assuming "user" as default userType if not specified
    await implementGoogleSignIn(context, "user");
  }

  // Implementation with fixed credential handling
  Future<void> implementGoogleSignIn(
      BuildContext context, String selectedUserType) async {
    final GoogleSignIn googleSignIn = GoogleSignIn(
      scopes: [
        'email',
      ],
    );
    try {
      showDialog(
          context: context,
          barrierDismissible: false,
          builder: (context) =>
              const Center(child: CircularProgressIndicator()));

      // Wait a moment to ensure sign out is complete
      await Future.delayed(const Duration(milliseconds: 500));

      // Trigger sign in flow
      final GoogleSignInAccount? gUser = await googleSignIn.signIn();

      if (gUser == null) {
        // Sign-in was canceled by user
        if (context.mounted) Navigator.of(context).pop();
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text('Sign-in canceled'),
            backgroundColor: Colors.orange,
          ));
        }
        return;
      }

      // Get authentication
      final GoogleSignInAuthentication gAuth = await gUser.authentication;

      // Check if tokens are valid
      if (gAuth.accessToken == null || gAuth.idToken == null) {
        if (context.mounted) Navigator.of(context).pop();
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text('Authentication failed: Invalid tokens'),
            backgroundColor: Colors.red,
          ));
        }
        return;
      }

      // Create credential
      final credential = GoogleAuthProvider.credential(
        accessToken: gAuth.accessToken!,
        idToken: gAuth.idToken!,
      );

      // Sign in with Firebase
      final userCredential =
          await FirebaseAuth.instance.signInWithCredential(credential);
      final User? user = userCredential.user;

      if (user == null) {
        if (context.mounted) Navigator.of(context).pop();
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text('Failed to sign in with Google'),
            backgroundColor: Colors.red,
          ));
        }
        return;
      }

      // Save user data to Firestore
      await FirebaseFirestore.instance.collection('users').doc(user.uid).set(
        {
          'name': user.displayName ?? "null",
          'email': user.email ?? "null",
          'userType': selectedUserType,
          'phone': user.phoneNumber ?? "null",
          'lastSignIn': FieldValue.serverTimestamp(),
        },
        SetOptions(merge: true), // Use merge to update existing documents
      );

      if (context.mounted) {
        Navigator.of(context).pop(); // Close loading dialog

        // Navigate to home page
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (context) => const HomePage()),
          (Route route) => false,
        );
      }
    } on FirebaseAuthException catch (e) {
      print('FirebaseAuthException: ${e.code} - ${e.message}');
      if (context.mounted) Navigator.of(context).pop();

      String errorMessage = 'Authentication failed';

      // Handle specific error codes
      switch (e.code) {
        case 'invalid-credential':
          errorMessage =
              'The credential is invalid or has expired. Please try again.';
          break;
        case 'user-disabled':
          errorMessage = 'This user account has been disabled.';
          break;
        case 'account-exists-with-different-credential':
          errorMessage =
              'An account already exists with the same email address.';
          break;
        case 'operation-not-allowed':
          errorMessage = 'Google sign-in is not enabled for this project.';
          break;
        case 'network-request-failed':
          errorMessage =
              'Network connection failed. Please check your internet connection.';
          break;
        default:
          errorMessage = e.message ?? 'Authentication failed';
      }

      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(errorMessage),
          backgroundColor: Colors.red,
        ));
      }
    } catch (e) {
      print('Error during Google Sign-In: $e');
      if (context.mounted) Navigator.of(context).pop();
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('An error occurred: $e'),
          backgroundColor: Colors.red,
        ));
      }
    }
  }

  // Email/Password Sign In Method
  Future<void> _signInWithEmailPassword() async {
    if (_emailController.text.isEmpty || _passwordController.text.isEmpty) {
      _showSnackBar('Please enter email and password');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      UserCredential userCredential;

      if (isLogin) {
        // Login existing user
        userCredential = await _auth.signInWithEmailAndPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text,
        );
        _showSnackBar('Successfully logged in');
        await FirebaseFirestore.instance
            .collection('logins')
            .doc("asjkbasckjasbcjklbasjkcBIY")
            .set(
          {
            "user_id": 1,
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timestamp": "2025-03-09 10:15:30",
            "ip_address": "192.168.1.100",
            "device_id": "device_123abc"
          },
        );
      } else {
        // Create new user
        if (_nameController.text.isEmpty) {
          _showSnackBar('Please enter your name');
          setState(() {
            _isLoading = false;
          });
          return;
        }

        userCredential = await _auth.createUserWithEmailAndPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text,
        );

        await userCredential.user
            ?.updateDisplayName(_nameController.text.trim());

        // Save new user data to Firestore
        await FirebaseFirestore.instance
            .collection('logins')
            .doc(userCredential.user!.uid)
            .set(
          {
            "user_id": 1,
            "latitude": 34.0522,
            "longitude": -118.2437,
            "timestamp": "2025-03-09 16:30:10",
            "ip_address": "192.168.1.102",
            "device_id": "device_123abc",
            "last_login_timestamp": "2025-03-09 14:20:45",
            "last_login_location": [40.7135, -74.0070]
          },
          SetOptions(merge: true),
        );

        _showSnackBar('Account created successfully');
      }

      if (userCredential.user != null) {
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (context) => const HomePage()),
          (Route route) => false,
        );
      }
    } catch (e) {
      String errorMessage = 'An error occurred';
      if (e is FirebaseAuthException) {
        switch (e.code) {
          case 'user-not-found':
            errorMessage = 'No user found with this email';
            break;
          case 'wrong-password':
            errorMessage = 'Wrong password';
            break;
          case 'email-already-in-use':
            errorMessage = 'Email is already in use';
            break;
          default:
            errorMessage = e.message ?? 'Authentication failed';
        }
      }
      _showSnackBar(errorMessage);
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Scaffold(
      backgroundColor: colors.surface,
      body: SafeArea(
        child: Stack(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 40),
                  _buildLogo(colors),
                  const SizedBox(height: 24),
                  _buildHeader(colors),
                  const SizedBox(height: 16),
                  Expanded(
                    child: FadeTransition(
                      opacity: _fadeAnimation,
                      child: SlideTransition(
                        position: _slideAnimation,
                        child: ScaleTransition(
                          scale: _scaleAnimation,
                          child: SingleChildScrollView(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.stretch,
                              children: [
                                if (!isLogin)
                                  _buildTextField(
                                      controller: _nameController,
                                      hint: 'Full Name',
                                      icon: Icons.person),
                                if (!isLogin) const SizedBox(height: 16),
                                if (!isLogin) _buildPhoneField(),
                                if (!isLogin) const SizedBox(height: 16),
                                _buildTextField(
                                    controller: _emailController,
                                    hint: 'Email',
                                    icon: Icons.email),
                                const SizedBox(height: 16),
                                _buildTextField(
                                    controller: _passwordController,
                                    hint: 'Password',
                                    icon: Icons.lock,
                                    isPassword: true),
                                const SizedBox(height: 24),
                                _buildActionButton(colors),
                                if (isLogin) _buildForgotPassword(colors),
                                const SizedBox(height: 24),
                                if (isLogin)
                                  const Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [Text("Or Login with")],
                                  ),
                                const SizedBox(height: 24),
                                if (isLogin)
                                  Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      _buildGoogleLogin(colors),
                                      const SizedBox(
                                          width:
                                              20), // Add some space between the buttons
                                      _buildGitHubLogin(colors),
                                    ],
                                  ),
                                _buildToggleButton(colors),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            if (_isLoading)
              Container(
                color: Colors.black.withOpacity(0.5),
                child: const Center(
                  child: CircularProgressIndicator(),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildLogo(ColorScheme colors) {
    return ScaleTransition(
      scale: _scaleAnimation,
      child: Container(
        width: 60,
        height: 60,
        decoration: BoxDecoration(
          color: colors.primary,
          shape: BoxShape.circle,
        ),
        child: Icon(
          Icons.school,
          color: colors.onPrimary,
          size: 40,
        ),
      ),
    );
  }

  Widget _buildHeader(ColorScheme colors) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            isLogin ? 'Welcome Back!' : 'Create Account',
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: colors.onSurface,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            isLogin
                ? 'Enter your email address and password to get access your account'
                : 'Please enter valid information to access your account.',
            style: TextStyle(
              fontSize: 14,
              color: colors.onSurface.withOpacity(0.7),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String hint,
    required IconData icon,
    bool isPassword = false,
  }) {
    return Builder(builder: (context) {
      final colors = Theme.of(context).colorScheme;
      return Container(
        decoration: BoxDecoration(
          color: colors.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: colors.onSurface.withOpacity(0.1)),
        ),
        child: TextField(
          onTap: () {
            setState(() {
              _emailController.text = "abhisheknaik2k20@gmail.com";
              _passwordController.text = "HelloWorld@1234";
            });
          },
          controller: controller,
          obscureText: isPassword && _obscureText,
          style: TextStyle(color: colors.onSurface),
          decoration: InputDecoration(
            hintText: hint,
            hintStyle: TextStyle(color: colors.onSurface.withOpacity(0.5)),
            prefixIcon: Icon(icon, color: colors.onSurface.withOpacity(0.5)),
            suffixIcon: isPassword
                ? IconButton(
                    icon: Icon(
                      _obscureText ? Icons.visibility : Icons.visibility_off,
                      color: colors.onSurface.withOpacity(0.5),
                    ),
                    onPressed: () {
                      setState(() {
                        _obscureText = !_obscureText;
                      });
                    },
                  )
                : null,
            border: InputBorder.none,
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          ),
        ),
      );
    });
  }

  Widget _buildPhoneField() {
    return Builder(builder: (context) {
      final colors = Theme.of(context).colorScheme;
      return Container(
        decoration: BoxDecoration(
          color: colors.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: colors.onSurface.withOpacity(0.1)),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.only(left: 16),
              child: Icon(
                Icons.phone,
                color: colors.onSurface.withOpacity(0.5),
              ),
            ),
            Expanded(
              child: TextField(
                controller: _phoneController,
                style: TextStyle(color: colors.onSurface),
                keyboardType: TextInputType.phone,
                decoration: InputDecoration(
                  hintText: 'Enter Phone',
                  hintStyle:
                      TextStyle(color: colors.onSurface.withOpacity(0.5)),
                  border: InputBorder.none,
                  contentPadding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                ),
              ),
            ),
          ],
        ),
      );
    });
  }

  Widget _buildForgotPassword(ColorScheme colors) {
    return Align(
      alignment: Alignment.center,
      child: TextButton(
        onPressed: () {
          // Implement forgot password logic
          _inAuthenticLogin();
        },
        child: Text(
          'IN_AUTHENTIC_LOGIN',
          style: TextStyle(color: colors.primary),
        ),
      ),
    );
  }

  Widget _buildActionButton(ColorScheme colors) {
    return ElevatedButton(
      onPressed: _isLoading ? null : _signInWithEmailPassword,
      style: ElevatedButton.styleFrom(
        foregroundColor: colors.onPrimary,
        backgroundColor: colors.primary,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        padding: const EdgeInsets.symmetric(vertical: 16),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            isLogin ? 'Login' : 'Create',
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          const SizedBox(width: 8),
          const Icon(Icons.arrow_forward, size: 18),
        ],
      ),
    );
  }

  Widget _buildGoogleLogin(ColorScheme colors) {
    return GestureDetector(
      onTap: _isLoading ? null : _signInWithGoogle,
      child: Container(
        width: 60,
        height: 60,
        decoration: BoxDecoration(
          color: Colors.grey.shade900,
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(
          child: SizedBox(
              width: 30,
              height: 30,
              child: Icon(
                FontAwesomeIcons.google,
                color: Colors.white,
                size: 30,
              )),
        ),
      ),
    );
  }

  Widget _buildGitHubLogin(ColorScheme colors) {
    return GestureDetector(
      onTap: _isLoading
          ? null
          : () async {
              // GitHub sign-in implementation would go here
              _showSnackBar('GitHub sign-in not implemented yet');
            },
      child: Container(
        width: 60,
        height: 60,
        decoration: BoxDecoration(
          color: Colors.grey[900],
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(
          child: SizedBox(
              width: 30,
              height: 30,
              child: Icon(
                FontAwesomeIcons.github,
                color: Colors.white,
                size: 30,
              )),
        ),
      ),
    );
  }

  Widget _buildToggleButton(ColorScheme colors) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          isLogin ? 'Don\'t have an account?' : 'Already have an account?',
          style: TextStyle(color: colors.onSurface.withOpacity(0.7)),
        ),
        TextButton(
          onPressed: _isLoading ? null : _toggleLoginSignup,
          style: ButtonStyle(
            overlayColor: WidgetStateProperty.resolveWith<Color?>(
              (states) {
                if (states.contains(WidgetState.pressed)) {
                  return Colors.transparent;
                }
                return null;
              },
            ),
          ),
          child: Text(
            isLogin ? 'Create account' : 'Login',
            style: TextStyle(
              color: colors.primary,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ],
    );
  }
}

// Placeholder class to avoid compilation errors - you should replace this with your actual homepage implementation
