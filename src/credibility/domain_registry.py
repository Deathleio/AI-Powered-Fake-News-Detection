import re
import urllib.parse
from typing import Dict, Any, Optional

# Verified domain authority database
# Tier 1 (90-100): International Wire Services, National Academies, Peer-Reviewed Journals
# Tier 2 (75-89): Established Major National Press, Public Broadcasters
# Tier 3 (50-74): Regional News, Digital Natives, Specialized Trade Media
# Satire (10-25): Known Satirical Publications (The Onion, Babylon Bee)
# Flagged (0-20): Documented Disinformation Outlets, Tabloid Farms, Conspiracy Blogs

DOMAIN_DATABASE = {
    # Tier 1: Wires & Peer-Reviewed Science
    "reuters.com": {"authority": 98, "type": "Tier 1 Wire Agency", "bias": "Least Biased", "verified": True},
    "apnews.com": {"authority": 98, "type": "Tier 1 Wire Agency", "bias": "Least Biased", "verified": True},
    "afp.com": {"authority": 96, "type": "Tier 1 Wire Agency", "bias": "Least Biased", "verified": True},
    "bloomberg.com": {"authority": 94, "type": "Financial News Wire", "bias": "Center", "verified": True},
    "nature.com": {"authority": 99, "type": "Peer-Reviewed Scientific Journal", "bias": "Pro-Science", "verified": True},
    "sciencemag.org": {"authority": 99, "type": "Peer-Reviewed Scientific Journal", "bias": "Pro-Science", "verified": True},
    "science.org": {"authority": 99, "type": "Peer-Reviewed Scientific Journal", "bias": "Pro-Science", "verified": True},
    "thelancet.com": {"authority": 99, "type": "Peer-Reviewed Medical Journal", "bias": "Pro-Science", "verified": True},
    "nejm.org": {"authority": 99, "type": "Peer-Reviewed Medical Journal", "bias": "Pro-Science", "verified": True},
    "nasa.gov": {"authority": 98, "type": "Government Scientific Agency", "bias": "Pro-Science", "verified": True},
    "who.int": {"authority": 97, "type": "Global Health Agency", "bias": "Pro-Science", "verified": True},
    "cdc.gov": {"authority": 97, "type": "Government Health Agency", "bias": "Pro-Science", "verified": True},

    # Tier 2: Established Mainstream & Public Broadcasters
    "bbc.com": {"authority": 92, "type": "Public Broadcaster", "bias": "Center", "verified": True},
    "bbc.co.uk": {"authority": 92, "type": "Public Broadcaster", "bias": "Center", "verified": True},
    "wsj.com": {"authority": 90, "type": "Major Financial Press", "bias": "Center-Right", "verified": True},
    "nytimes.com": {"authority": 89, "type": "Major National Press", "bias": "Center-Left", "verified": True},
    "washingtonpost.com": {"authority": 88, "type": "Major National Press", "bias": "Center-Left", "verified": True},
    "theguardian.com": {"authority": 87, "type": "Major National Press", "bias": "Center-Left", "verified": True},
    "economist.com": {"authority": 92, "type": "International Affairs Weekly", "bias": "Center", "verified": True},
    "ft.com": {"authority": 93, "type": "Major Financial Press", "bias": "Center", "verified": True},
    "npr.org": {"authority": 90, "type": "Public Radio Broadcaster", "bias": "Center-Left", "verified": True},
    "pbs.org": {"authority": 91, "type": "Public Television Broadcaster", "bias": "Center", "verified": True},
    "thehindu.com": {"authority": 86, "type": "Major National Press", "bias": "Center", "verified": True},
    "indianexpress.com": {"authority": 85, "type": "Major National Press", "bias": "Center", "verified": True},
    "lemonde.fr": {"authority": 89, "type": "Major National Press", "bias": "Center-Left", "verified": True},

    # Fact-Checking Organizations
    "politifact.com": {"authority": 95, "type": "Certified Fact-Checker", "bias": "Least Biased", "verified": True},
    "snopes.com": {"authority": 93, "type": "Fact-Checking Organization", "bias": "Least Biased", "verified": True},
    "factcheck.org": {"authority": 95, "type": "Certified Fact-Checker", "bias": "Least Biased", "verified": True},
    "fullfact.org": {"authority": 94, "type": "Certified Fact-Checker", "bias": "Least Biased", "verified": True},

    # Satire & Parody Outlets
    "theonion.com": {"authority": 15, "type": "Satire / Parody", "bias": "Satire", "verified": False},
    "babylonbee.com": {"authority": 15, "type": "Satire / Parody", "bias": "Satire", "verified": False},
    "thedailymash.co.uk": {"authority": 15, "type": "Satire / Parody", "bias": "Satire", "verified": False},
    "newsthump.com": {"authority": 15, "type": "Satire / Parody", "bias": "Satire", "verified": False},

    # Documented Disinformation / Sensationalist Domains
    "infowars.com": {"authority": 8, "type": "Conspiracy / Disinformation", "bias": "Extreme Right", "verified": False},
    "worldnewsdailyreport.com": {"authority": 5, "type": "Fabricated Hoaxes", "bias": "Fake News", "verified": False},
    "beforeitsnews.com": {"authority": 8, "type": "Unvetted Conspiracy Blog", "bias": "Extreme", "verified": False},
    "naturalnews.com": {"authority": 10, "type": "Medical Disinformation", "bias": "Conspiracy", "verified": False},
    "breitbart.com": {"authority": 35, "type": "Hyperpartisan Outlet", "bias": "Right", "verified": False}
}

def extract_domain_from_url(url_or_text: str) -> Optional[str]:
    """Extracts root domain (e.g. 'bbc.com') from a URL or source text."""
    if not url_or_text:
        return None
    
    # Try parsing as URL
    try:
        parsed = urllib.parse.urlparse(url_or_text)
        netloc = parsed.netloc or parsed.path
        netloc = re.sub(r'^www\.', '', netloc.lower())
        netloc = netloc.split('/')[0].split(':')[0]
        if '.' in netloc:
            return netloc
    except Exception:
        pass
    
    # Search for domain patterns inside text
    match = re.search(r'\b([a-zA-Z0-9-]+\.(?:com|org|gov|edu|net|co\.uk|int|io|in|fr|de))\b', url_or_text.lower())
    if match:
        return match.group(1)
        
    return None

def evaluate_publisher_credibility(url_or_domain: Optional[str]) -> Dict[str, Any]:
    """
    Evaluates publisher credibility against global news authority database.
    """
    domain = extract_domain_from_url(url_or_domain) if url_or_domain else None
    
    if not domain or domain not in DOMAIN_DATABASE:
        # Check partial root domain match (e.g. "news.bbc.co.uk" -> "bbc.co.uk")
        matched_info = None
        if domain:
            for known_dom, info in DOMAIN_DATABASE.items():
                if domain.endswith("." + known_dom) or domain == known_dom:
                    matched_info = info
                    domain = known_dom
                    break
                    
        if not matched_info:
            return {
                "domain": domain or "Unknown Source",
                "authority_score": 50.0,
                "publisher_type": "Unregistered / Independent Source",
                "bias_rating": "Unknown / Unrated",
                "is_verified_journalistic": False,
                "is_satire": False,
                "is_flagged_disinfo": False
            }
        else:
            info = matched_info
    else:
        info = DOMAIN_DATABASE[domain]

    is_satire = "Satire" in info["type"]
    is_flagged = info["authority"] < 25 and not is_satire

    return {
        "domain": domain,
        "authority_score": float(info["authority"]),
        "publisher_type": info["type"],
        "bias_rating": info["bias"],
        "is_verified_journalistic": info["verified"],
        "is_satire": is_satire,
        "is_flagged_disinfo": is_flagged
    }
