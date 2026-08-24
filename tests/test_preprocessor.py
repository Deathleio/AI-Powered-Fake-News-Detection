import unittest
import pandas as pd
from src.data.preprocessor import sanitize_wire_leakage, fuse_title_body, TextPreprocessor

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

if __name__ == '__main__':
    unittest.main()
