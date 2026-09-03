import re
import json
import html
import urllib.parse
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

def clean_and_normalize_url(url: str) -> str:
    """
    Cleans tracking parameters (utm_*, fbclid, ref) and normalizes scheme.
    """
    if not url:
        return ""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
        
    try:
        parsed = urllib.parse.urlparse(url)
        # Strip tracking query parameters
        query_params = urllib.parse.parse_qsl(parsed.query)
        cleaned_params = [
            (k, v) for (k, v) in query_params 
            if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid", "ref", "source", "mc_cid"}
        ]
        new_query = urllib.parse.urlencode(cleaned_params)
        normalized = urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            "" # strip fragment/anchor
        ))
        return normalized
    except Exception:
        return url

def extract_article_from_url(url: str, timeout: float = 6.0) -> Dict[str, Any]:
    """
    Robust 4-tier web article extraction pipeline:
      1. Resilient HTTP retrieval with desktop browser headers & cookie jar.
      2. Structured JSON-LD schema.org/NewsArticle extraction.
      3. OpenGraph & Twitter Card rich metadata extraction.
      4. Semantic DOM parsing via BeautifulSoup targeting <article>, <main>, and <p>.
    """
    normalized_url = clean_and_normalize_url(url)
    if not normalized_url:
        return {
            "success": False,
            "error": "Invalid or empty URL provided.",
            "url": url,
            "domain": "",
            "title": "",
            "text": ""
        }

    domain = urllib.parse.urlparse(normalized_url).netloc.replace("www.", "").lower()
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Sec-Ch-Ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1'
    }

    try:
        resp = session.get(normalized_url, headers=headers, timeout=timeout, allow_redirects=True)
        if resp.status_code == 404:
            return {"success": False, "error": "Article link not found (HTTP 404).", "url": normalized_url, "domain": domain, "title": "", "text": ""}
        if resp.status_code >= 400:
            return {"success": False, "error": f"Target server returned HTTP {resp.status_code}.", "url": normalized_url, "domain": domain, "title": "", "text": ""}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Connection timed out while trying to reach news outlet.", "url": normalized_url, "domain": domain, "title": "", "text": ""}
    except requests.exceptions.SSLError:
        # Fallback with relaxed SSL verification if news site has self-signed/expired cert
        try:
            resp = session.get(normalized_url, headers=headers, timeout=timeout, allow_redirects=True, verify=False)
        except Exception as e:
            return {"success": False, "error": f"SSL connection error: {str(e)}", "url": normalized_url, "domain": domain, "title": "", "text": ""}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch URL: {str(e)}", "url": normalized_url, "domain": domain, "title": "", "text": ""}

    final_url = resp.url or normalized_url
    domain = urllib.parse.urlparse(final_url).netloc.replace("www.", "").lower()

    # Intelligent charset handling
    if resp.encoding is None or resp.encoding.lower() == 'iso-8859-1':
        resp.encoding = resp.apparent_encoding or 'utf-8'

    soup = BeautifulSoup(resp.text, 'html.parser')

    title = ""
    author = ""
    published_date = ""
    body_paragraphs: list = []

    # ----------------------------------------------------
    # TIER 1: JSON-LD Structured Data (Gold Standard)
    # ----------------------------------------------------
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            raw_json = script.string
            if not raw_json:
                continue
            data = json.loads(raw_json)
            nodes = data if isinstance(data, list) else [data]
            for item in nodes:
                graph = item.get('@graph', [item]) if isinstance(item, dict) else [item]
                for node in graph:
                    if not isinstance(node, dict):
                        continue
                    node_type = str(node.get('@type', ''))
                    if any(t in node_type for t in ['NewsArticle', 'Article', 'BlogPosting', 'Report', 'WebPage']):
                        if not title and node.get('headline'):
                            title = str(node.get('headline'))
                        if not published_date and (node.get('datePublished') or node.get('dateCreated')):
                            published_date = str(node.get('datePublished') or node.get('dateCreated'))
                        if not author and node.get('author'):
                            auth_val = node.get('author')
                            if isinstance(auth_val, list) and auth_val:
                                author = auth_val[0].get('name', '') if isinstance(auth_val[0], dict) else str(auth_val[0])
                            elif isinstance(auth_val, dict):
                                author = auth_val.get('name', '')
                            elif isinstance(auth_val, str):
                                author = auth_val
                        if not body_paragraphs and node.get('articleBody') and isinstance(node.get('articleBody'), str):
                            raw_body = node.get('articleBody').strip()
                            if len(raw_body) > 100:
                                body_paragraphs.append(raw_body)
        except Exception:
            continue

    # ----------------------------------------------------
    # TIER 2: OpenGraph & Twitter Metadata Fallback
    # ----------------------------------------------------
    if not title:
        og_t = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'twitter:title'})
        if og_t and og_t.get('content'):
            title = og_t['content']
        elif soup.title and soup.title.string:
            title = soup.title.string

    if not published_date:
        pub_meta = soup.find('meta', property='article:published_time') or soup.find('meta', attrs={'name': 'pubdate'}) or soup.find('meta', attrs={'name': 'publish-date'})
        if pub_meta and pub_meta.get('content'):
            published_date = pub_meta['content']

    if not author:
        auth_meta = soup.find('meta', attrs={'name': 'author'}) or soup.find('meta', property='article:author')
        if auth_meta and auth_meta.get('content'):
            author = auth_meta['content']

    # Clean brand suffixes from title (e.g. "Headline | BBC News", " - Reuters")
    if title:
        title = html.unescape(title).strip()
        title = re.sub(r'\s*[-|–—]\s*(?:BBC|Reuters|CNN|The Guardian|The New York Times|AP News|Associated Press|Fox News|NPR|CNBC|The Wall Street Journal|WSJ|Forbes|USA Today).*$', '', title, flags=re.I).strip()

    # ----------------------------------------------------
    # TIER 3: Semantic DOM Extraction via BeautifulSoup
    # ----------------------------------------------------
    if not body_paragraphs:
        # Strip script, style, nav, and peripheral noise
        for unwanted in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'noscript', 'svg', 'button', 'iframe']):
            unwanted.decompose()

        # Target semantic article container if present
        container = (
            soup.find('article') or 
            soup.find('main') or 
            soup.find('div', class_=re.compile(r'(article[-_]body|story[-_]body|post[-_]content|entry[-_]content|article[-_]content)', re.I)) or 
            soup
        )

        BOILERPLATE = [
            'cookie policy', 'privacy policy', 'terms of service', 'all rights reserved',
            'sign up for our', 'subscribe now', 'click here to read', 'advertisement',
            'follow us on', 'read more:', 'related articles:', 'share this article'
        ]

        for p in container.find_all('p'):
            p_text = p.get_text(separator=' ', strip=True)
            p_clean = html.unescape(p_text)
            p_clean = re.sub(r'\s+', ' ', p_clean).strip()
            
            # Filter low-value text, copyright notices, and cookie policies
            if len(p_clean) > 35 and not any(bp in p_clean.lower() for bp in BOILERPLATE):
                body_paragraphs.append(p_clean)

    # ----------------------------------------------------
    # TIER 4: Fallback to Meta Description
    # ----------------------------------------------------
    final_text = " ".join(body_paragraphs)
    if len(final_text) < 80:
        desc_tag = soup.find('meta', property='og:description') or soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'name': 'twitter:description'})
        if desc_tag and desc_tag.get('content'):
            final_text = html.unescape(desc_tag['content']).strip()

    words = final_text.split()
    word_count = len(words)
    reading_time = max(1, round(word_count / 200)) if word_count > 0 else 0

    success = bool(title or final_text)
    error_msg = None if success else "Could not extract readable article text. The page may require JavaScript or authentication."

    return {
        "success": success,
        "error": error_msg,
        "url": final_url,
        "domain": domain,
        "title": title,
        "author": author,
        "published_date": published_date,
        "text": final_text[:5000],
        "word_count": word_count,
        "reading_time_min": reading_time
    }

