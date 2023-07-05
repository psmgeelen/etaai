import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:url_launcher/url_launcher.dart';

final Uri _url = Uri.parse('https://github.com/psmgeelen/etaai');

void main() {
  runApp(const MyApp());
}

Future<void> _launchUrl() async {
  if (!await launchUrl(_url)) {
    throw Exception('Could not launch $_url');
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        theme: ThemeData(
          textTheme: GoogleFonts.robotoMonoTextTheme(),
        ),
        home: Home());
  }
}

class Home extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromRGBO(49, 49, 58, 1),
      body: Center(
        child: Column(
          children: [
            Image.asset(
              'assets/images/logo.jpeg',
              alignment: Alignment.topCenter,
              height: MediaQuery.of(context).size.height * 0.3,
            ),
            const SizedBox(
              height: 80,
            ),
            SizedBox(
              width: MediaQuery.of(context).size.width * 0.6,
              child: RichText(
                textAlign: TextAlign.center,
                text: TextSpan(
                  style: const TextStyle(
                    color: Color(0xfff0f0f0),
                    fontSize: 22,
                  ),
                  children: [
                    const TextSpan(
                        style: TextStyle(height: 1.5),
                        text: 'Welcome to the landing page for η.ai. '
                            'The project is still under development '
                            'and will take some time to complete. '
                            'If you want to follow the development, please visit the '),
                    const WidgetSpan(
                      child: FaIcon(
                        FontAwesomeIcons.github,
                        color: Color(0xfff0f0f0),
                      ),
                    ),
                    TextSpan(
                      text: ' repo',
                      style: const TextStyle(
                        color: Color(0xfff16c40),
                      ),
                      recognizer: TapGestureRecognizer()..onTap = _launchUrl,
                    ),
                    const TextSpan(text: '!'),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
