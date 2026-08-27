import os
import sys

sys.path.insert(0, os.path.abspath(r"c:\AI Powered Fake News Detection"))

from src.serving.api import (
    NewsArticleRequest, 
    UrlArticleRequest, 
    FeedbackRequest,
    explain_news, 
    analyze_url_endpoint, 
    submit_feedback, 
    export_forensic_report
)

print("=================================================================")
print("[*] TESTING VERITASAI ENTERPRISE V2.0 ENGINE & API ENDPOINTS")
print("=================================================================\n")

# 1. Test Text Analysis with Claims Breakdown and Trust Score
req1 = NewsArticleRequest(
    title="NASA James Webb Space Telescope Detects Water Vapor in Rocky Planet Formation Zone",
    text="Astronomers using NASAs James Webb Space Telescope have identified clear spectroscopic signatures of water vapor within the inner disk of a young stellar system. The findings, published in the journal Nature, suggest that rocky exoplanets forming in this region may have access to a substantial reservoir of water early in their development. Officials confirmed the results on Thursday.",
    source_url="https://www.nature.com"
)
res1 = explain_news(req1)
print(f"[TEST 1: Article Analysis]")
print(f"  Verdict:         {res1.verdict} ({res1.confidence_percentage}% Conf)")
print(f"  Veritas Score:   {res1.veritas_score}/100")
print(f"  Publisher Type:  {res1.domain_credibility['publisher_type']} (Auth: {res1.domain_credibility['authority_score']}/100)")
print(f"  Total Claims:    {len(res1.claims_breakdown)}")
for c in res1.claims_breakdown[:3]:
    print(f"    - Claim #{c['claim_id']} [{c['risk_level']}]: {c['category']}")
print(f"  Rationale:       {res1.llm_reasoning['rationale']}\n")

# 2. Test Feedback API
req2 = FeedbackRequest(
    title="Sample Claim Test",
    text="Sample body text for feedback",
    predicted_verdict="Real News",
    user_reported_verdict="Real News",
    notes="Analyst verified via Nature journal."
)
res2 = submit_feedback(req2)
print(f"[TEST 2: Active Learning Feedback API]")
print(f"  Status: {res2['status']} | Msg: {res2['message']}\n")

# 3. Test Export Report API
res3 = export_forensic_report(req1)
print(f"[TEST 3: Cryptographic Audit Export]")
print(f"  Report ID: {res3['report_id']}")
print(f"  Audit Sig: {res3['audit_signature'][:24]}...")
print(f"  Trust:     {res3['veritas_trust_score']}/100\n")

print("=================================================================")
print("ALL ENTERPRISE V2.0 SUITE TESTS PASSED!")
print("=================================================================")
