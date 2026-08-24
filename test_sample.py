import sys
import os
import argparse

# UTF-8 stdout configuration
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.serving.api import NewsArticleRequest, explain_news

DEMO_SAMPLES = [
    {
        "category": "Verified Mainstream News (Factual)",
        "expected": "Real News",
        "title": "Federal Reserve Holds Benchmark Interest Rates Steady Amid Stable Economic Growth",
        "text": "The Federal Reserve announced on Wednesday that it will maintain its benchmark interest rate within the current target range following a unanimous vote by the Federal Open Market Committee. Central bank officials cited continuing job growth, steady consumer spending, and moderate inflation figures in their official policy statement released in Washington."
    },
    {
        "category": "Viral Sensational Clickbait / Fake News",
        "expected": "Fake News",
        "title": "SHOCKING BOMBSHELL: Secret Globalist Plot Leaked To Ban All Cash And Confiscate Savings By Next Week [VIDEO]",
        "text": "UNBELIEVABLE! Top secret government whistleblowers have exposed an explosive classified document proving corrupt globalist elites are orchestrating a total financial blackout to seize your private bank accounts! Mainstream media refuses to report this terrifying scheme. Watch the emergency video before censors take it down!"
    },
    {
        "category": "Scientific Research Dispatch (Real)",
        "expected": "Real News",
        "title": "James Webb Space Telescope Detects Water Vapor in Rocky Planet Formation Zone",
        "text": "Astronomers using NASAs James Webb Space Telescope have identified clear spectroscopic signatures of water vapor within the inner disk of a young stellar system. The findings, published in the journal Nature, suggest that rocky exoplanets forming in this region may have access to a substantial reservoir of water early in their development."
    },
    {
        "category": "Medical / Health Disinformation (Fake)",
        "expected": "Fake News",
        "title": "MIRACLE CURE EXPOSED: Big Pharma Panic As Secret Ancient Root Cures All Disease Overnight [MUST SEE]",
        "text": "Doctors are STUNNED and corrupt pharmaceutical executives are in a panic! This 100% natural ancient herbal remedy is being suppressed because it completely reverses aging and cures every chronic condition in just 24 hours. The medical establishment does not want you to know the truth!"
    }
]

def format_tokens(token_list):
    return ", ".join([t['token'] + " (" + str(t['weight']) + ")" for t in token_list])

def test_article(title: str, text: str):
    req = NewsArticleRequest(title=title, text=text)
    res = explain_news(req)
    
    print("\n" + "=" * 65)
    print("INPUT ARTICLE:")
    print(f"  Title: {title}")
    print(f"  Body:  {text[:180]}...")
    print("-" * 65)
    print("MODEL PREDICTION:")
    
    status_icon = "[FAKE / CLICKBAIT]" if res.is_fake else "[REAL / FACTUAL]"
        
    print(f"  Verdict:              {status_icon} {res.verdict}")
    print(f"  Confidence:           {res.confidence_percentage}%")
    print(f"  Fake Probability:     {res.fake_probability}")
    
    if res.is_fake and res.fake_indicators:
        print(f"  Red-Flag Keywords:    {format_tokens(res.fake_indicators[:5])}")
    elif not res.is_fake and res.real_indicators:
        print(f"  Credibility Keywords: {format_tokens(res.real_indicators[:5])}")
        
    print(f"  AI Rationale:         {res.llm_reasoning.get('rationale')}")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Test Fake News Detection Model on custom inputs")
    parser.add_argument("--title", type=str, default=None, help="News headline/title")
    parser.add_argument("--text", type=str, default=None, help="News article body")
    parser.add_argument("--demo", action="store_true", help="Run full benchmark test suite on 4 realistic sample articles")
    
    args = parser.parse_args()
    
    if args.title or args.text:
        test_article(args.title or "", args.text or "")
    else:
        print("\n=======================================================")
        print("[*] RUNNING DEMO SAMPLES THROUGH TRAINED MODEL")
        print("=======================================================")
        for i, sample in enumerate(DEMO_SAMPLES, 1):
            print(f"\n[Test Case {i}/4] Type: {sample['category']} (Expected: {sample['expected']})")
            test_article(sample["title"], sample["text"])

if __name__ == '__main__':
    main()
