import unittest
import pandas as pd
from src.data.preprocessor import sanitize_wire_leakage, fuse_title_body, TextPreprocessor, extract_stylistic_features

class TestPreprocessor(unittest.TestCase):
    def test_sanitize_wire_leakage(self):
        text1 = "WASHINGTON (Reuters) - The U.S. Senate passed a landmark bill today."
        cleaned1 = sanitize_wire_leakage(text1)
        self.assertNotIn("Reuters", cleaned1)
        self.assertTrue(cleaned1.startswith("The U.S. Senate"))
        
        text2 = "Breaking news from Europe - Breitbart"
        cleaned2 = sanitize_wire_leakage(text2)
        self.assertNotIn("Breitbart", cleaned2)
        
        text3 = "SHOCKING VIDEO [VIDEO] MUST SEE https://news.com/video"
        cleaned3 = sanitize_wire_leakage(text3)
        self.assertNotIn("[VIDEO]", cleaned3)
        self.assertIn("[URL]", cleaned3)

    def test_fuse_title_body(self):
        title = "Congress Passes Relief Package"
        body = "Lawmakers voted overwhelmingly on Friday."
        fused = fuse_title_body(title, body, title_repeat=2)
        self.assertIn(title, fused)
        self.assertIn(body, fused)
        self.assertEqual(fused.count(title), 2)

    def test_text_preprocessor_dataframe(self):
        df = pd.DataFrame({
            'title': ["Test Title 1", None],
            'text': ["Test Body 1", "Test Body 2"]
        })
        tp = TextPreprocessor(title_repeat=1)
        res = tp.transform(df)
        self.assertEqual(len(res), 2)
        self.assertIn("Test Title 1", res[0])
        self.assertIn("Test Body 2", res[1])

    def test_extract_stylistic_features(self):
        # Test all-caps and sensationalist alarm
        features = extract_stylistic_features(
            "THE WORLD IS ON FIRE",
            "AUSTRALIA BUSHFIRE HAS TAKEN THE LIFE OF TRUMP WHO WAS DANCING WITH NETANYAHU"
        )
        self.assertTrue(features["is_all_caps_title"])
        self.assertTrue(features["is_all_caps_body"])
        self.assertGreater(features["stylistic_fake_risk"], 0.6)
        self.assertIn("world is on fire", [k.lower() for k in features["sensational_keywords"]])

        # Test authentic news with journalistic attribution
        real_features = extract_stylistic_features(
            "Federal Reserve Holds Benchmark Interest Rates Steady",
            "The Federal Reserve announced on Wednesday that benchmark rates will remain steady, according to official statements."
        )
        self.assertFalse(real_features["is_all_caps_title"])
        self.assertGreater(real_features["attribution_score"], 0)
        self.assertLess(real_features["stylistic_fake_risk"], 0.2)

if __name__ == '__main__':
    unittest.main()
