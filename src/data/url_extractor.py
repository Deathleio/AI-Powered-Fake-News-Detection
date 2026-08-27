import re
import html
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional

def extract_article_from_url(url: str, timeout: float = 4.0) -> Dict[str, Any]:
    """
    Fetches and parses a web article URL, extracting:
      - Clean Title (from <title> or <meta property="og:title">)
      - Main Content / Body text (from <article>, <p> tags, or <meta description>)
      - Publisher Domain & Canonical URL
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 VeritasAI-Fetcher/1.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_html = response.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return {
            "success": False,
            "error": f"Unable to reach or fetch URL: {str(e)}",
            "url": url,
            "title": "",
            "text": "",
            "domain": ""
        }

    # Extract Domain
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc.replace("www.", "")

    # 1. Extract Title
    title = ""
    og_title_match = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']', raw_html, re.I)
    if not og_title_match:
        og_title_match = re.search(r'<meta\s+name=["\']twitter:title["\']\s+content=["\']([^"\']+)["\']', raw_html, re.I)
        
    if og_title_match:
        title = og_title_match.group(1)
    else:
        title_tag_match = re.search(r'<title[^>]*>(.*?)</title>', raw_html, re.I | re.S)
        if title_tag_match:
            title = title_tag_match.group(1)

    title = html.unescape(title).strip()
    # Clean trailing brand suffixes like " | BBC News", " - Reuters"
    title = re.sub(r'\s*[-|–—]\s*(?:BBC|Reuters|CNN|The Guardian|The New York Times|AP News|Fox News).*$', '', title, flags=re.I).strip()

    # 2. Extract Body Content
    # Strip script and style tags
    clean_html = re.sub(r'<(script|style|nav|header|footer|aside)[^>]*>.*?</\1>', ' ', raw_html, flags=re.I | re.S)
    
    # Try finding <article> block first
    article_match = re.search(r'<article[^>]*>(.*?)</article>', clean_html, re.I | re.S)
    search_scope = article_match.group(1) if article_match else clean_html
    
    # Extract <p> tags
    p_tags = re.findall(r'<p[^>]*>(.*?)</p>', search_scope, re.I | re.S)
    paragraphs = []
    for p in p_tags:
        clean_p = re.sub(r'<.*?>', ' ', p)
        clean_p = html.unescape(clean_p)
        clean_p = re.sub(r'\s+', ' ', clean_p).strip()
        if len(clean_p) > 25 and not any(skip in clean_p.lower() for skip in ['cookie policy', 'privacy notice', 'all rights reserved', 'sign up for']):
            paragraphs.append(clean_p)
            
    body_text = " ".join(paragraphs)
    
    # Fallback to meta description if body text is sparse
    if len(body_text) < 100:
        desc_match = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\']([^"\']+)["\']', raw_html, re.I)
        if not desc_match:
            desc_match = re.search(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']', raw_html, re.I)
        if desc_match:
            body_text = html.unescape(desc_match.group(1)).strip()

    return {
        "success": bool(title or body_text),
        "error": None if (title or body_text) else "Failed to parse text from web page.",
        "url": url,
        "domain": domain,
        "title": title,
        "text": body_text[:4000] # Cap text for efficient downstream processing
    }
