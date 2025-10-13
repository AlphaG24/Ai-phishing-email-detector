import re
from bs4 import BeautifulSoup
import email
from email import policy
from email.parser import BytesParser
import io
import idna # Library for Punycode conversion

# --- Helper Functions (No Change) ---

def count_urls(text):
    """Counts the number of URLs in a given string of text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = re.findall(url_pattern, text)
    return len(urls)

def check_suspicious_keywords(text):
    """Checks for the presence of suspicious keywords often used in phishing."""
    suspicious_keywords = [
        'urgent', 'action required', 'verify', 'account suspended',
        'security alert', 'password reset', 'click here', 'confirm', 'disabled'
    ]
    text_lower = text.lower()
    for keyword in suspicious_keywords:
        if keyword in text_lower:
            return 1
    return 0
    
def check_suspicious_title(text):
    """Checks if the email subject/title contains suspicious patterns (e.g., empty or generic)."""
    title_pattern = re.compile(r'^(re|fw):\s*$', re.IGNORECASE)
    if not text.strip() or title_pattern.search(text):
        return 1
    return 0

def has_hidden_images(html_content):
    """Conceptual check for the presence of hidden images (e.g., 1x1 tracking pixels)."""
    hidden_image_pattern = re.compile(
        r'<img[^>]+(width=["\']?1["\']?|height=["\']?1["\']?|style=["\'][^"\']*(width:\s*1px|height:\s*1px|display:\s*none|visibility:\s*hidden)[^"\']*["\'])',
        re.IGNORECASE
    )
    if hidden_image_pattern.search(html_content):
        return 1
    return 0
    
def check_url_suspiciousness(text):
    """
    Checks for suspicious URL patterns within the text.
    Example patterns: IPs in URL, excessive use of dots in domain, non-standard port numbers.
    """
    url_pattern = re.compile(r'(?:https?:\/\/(?:[0-9]{1,3}\.){3}[0-9]{1,3})|(\.{3,})|(:[0-9]{4,})')
    if url_pattern.search(text):
        return 1
    return 0
    
def calculate_html_text_ratio(raw_content):
    """
    Calculates a basic HTML to visible text ratio (a sign of hidden content).
    Returns 1 if the ratio is excessively high (suggesting hidden or complex HTML), otherwise 0.
    """
    try:
        soup = BeautifulSoup(raw_content, 'html.parser')
        visible_text = soup.get_text()
        html_length = len(raw_content)
        
        if html_length > 1000 and len(visible_text) < 100:
            return 1
        return 0
    except:
        return 0

def count_misspellings(text):
    """Simple conceptual check for common spelling mistakes in financial/security terms."""
    common_phish_typos = ['paypaI', 'googIe', 'verifiy', 'securty', 'updte']
    text_lower = text.lower()
    for typo in common_phish_typos:
        if typo in text_lower:
            return 1
    return 0

def check_header_spoofing(raw_email_content):
    """
    Conceptual check for header anomalies (e.g., missing security headers like SPF/DKIM).
    """
    if re.search(r'Authentication-Results:.*(spf=fail|dkim=fail)', raw_email_content, re.IGNORECASE):
        return 1
    
    from_match = re.search(r'From:.*<(.*)>', raw_email_content)
    return_path_match = re.search(r'Return-Path:.*<(.*)>', raw_email_content)
    
    if from_match and return_path_match:
        from_domain = from_match.group(1).split('@')[-1]
        return_domain = return_path_match.group(1).split('@')[-1]
        
        if from_domain.lower() != return_domain.lower():
            return 1
            
    return 0

# --- DOMAIN AGE SIMULATION (No Change) ---
def extract_root_domain(url):
    """Simple method to extract root domain (conceptual, not production ready)."""
    try:
        if 'http' not in url:
            url = 'http://' + url
        root_domain = re.search(r'https?://([^\/]+)', url).group(1)
        
        # Simple Punycode/IDNA decoding check for detection
        if root_domain.startswith('xn--'):
             return root_domain # Flag as potential homoglyph/punycode
        
        # Remove subdomains
        parts = root_domain.split('.')
        return f"{parts[-2]}.{parts[-1]}"
    except:
        return None

def get_domain_age_days(raw_email_content):
    """
    Simulates checking the age of a domain found in the email content.
    Returns 1 if a suspicious domain age is detected (< 60 days).
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = re.findall(url_pattern, raw_email_content)
    
    if not urls:
        return 0

    domain = extract_root_domain(urls[0])

    if domain:
        if 'verify-portal.net' in domain.lower() or 'secure-login.com' in domain.lower():
            return 1 
        elif 'amazonservices.com' in domain.lower() or 'google.com' in domain.lower():
            return 0
        return 0 
    return 0
# --- END DOMAIN AGE ---

# --- NEW FEATURE: TYPOSQUATTING (Homoglyph/Punycode) ---
def check_for_typosquatting(raw_email_content):
    """
    Checks for high-risk typosquatting indicators: Punycode (IDN Homograph attacks).
    Returns 1 if a suspicious URL is found that uses homoglyphs.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = re.findall(url_pattern, raw_email_content)
    
    for url in urls:
        try:
            # Extract domain part
            domain_match = re.search(r'https?://([^\/]+)', url)
            if not domain_match:
                continue

            domain = domain_match.group(1)
            
            # 1. Punycode check (A common trick is using Cyrillic 'a' to spoof 'apple.com' as 'аррle.com' -> xn--рre.com)
            if domain.startswith('xn--'):
                # Attempt to decode it to see the homoglyphs
                decoded_domain = idna.decode(domain)
                # Simple rule: if the decoded domain looks like a major brand
                if 'apple' in decoded_domain or 'google' in decoded_domain or 'bank' in decoded_domain:
                    return 1
            
            # 2. Character substitution check (e.g., micr0soft instead of microsoft - handled conceptually here)
            if 'micr0s0ft' in domain.lower() or 'paypai' in domain.lower():
                 return 1
                 
        except:
            continue
            
    return 0

def check_domain_reputation(url):
    """Check if domain has poor reputation indicators"""
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top']
    new_tlds = ['.online', '.club', '.site', '.website']
    
    for tld in suspicious_tlds:
        if tld in url.lower():
            return 1
    return 0

def check_url_obfuscation(text):
    """Detect URL encoding tricks and obfuscation"""
    # Check for hex encoding
    if re.search(r'%[0-9a-fA-F]{2}', text):
        return 1
    # Check for excessive special characters in URLs
    url_pattern = re.compile(r'https?://[^\s]+')
    urls = re.findall(url_pattern, text)
    for url in urls:
        if len(re.findall(r'[!@#$%^&*()+=]', url)) > 3:
            return 1
    return 0

def check_email_authentication(raw_email):
    """Check SPF, DKIM, DMARC headers (conceptual)"""
    headers = raw_email.lower()
    
    # Check for authentication headers
    auth_indicators = 0
    if 'authentication-results' in headers:
        auth_indicators += 1
    if 'received-spf' in headers:
        auth_indicators += 1
    if 'dkim-signature' in headers:
        auth_indicators += 1
    
    # More headers = more legitimate
    return 0 if auth_indicators >= 2 else 1

def check_urgency_language(text):
    """Detect urgent language patterns"""
    urgency_patterns = [
        r'immediate\s+action', r'within\s+\d+\s+hours', 
        r'account\s+closure', r'limited\s+time',
        r'act\s+now', r'don\'t\s+delay', r'last\s+chance'
    ]
    text_lower = text.lower()
    for pattern in urgency_patterns:
        if re.search(pattern, text_lower):
            return 1
    return 0

def check_grammar_quality(text):
    """Simple grammar and professionalism check"""
    # Count capitalization errors
    words = text.split()
    if len(words) < 10:  # Too short for proper analysis
        return 0
    
    capital_errors = sum(1 for word in words if word.isupper() and len(word) > 3)
    if capital_errors > len(words) * 0.1:  # More than 10% words in ALL CAPS
        return 1
    return 0

def analyze_multiple_emails(emails_list):
    """Bulk analysis for multiple emails"""
    results = []
    for email_text in emails_list:
        features = get_advanced_features(email_text)
        # Add your prediction logic here
        results.append({
            'email_preview': email_text[:100] + '...',
            'features': features,
            'prediction': 'phishing' if sum(features.values()) > 5 else 'legitimate'
        })
    return results
# --- END NEW FEATURE ---


def get_advanced_features(text):
    """
    Extracts all advanced features from the text (now 14 features)
    """
    if not isinstance(text, str):
        text = ""
    
    features = {}
    
    # Original 10 features
    features['num_urls'] = count_urls(text)
    features['has_suspicious_keywords'] = check_suspicious_keywords(text)
    features['has_suspicious_title'] = check_suspicious_title(text)
    features['has_hidden_images'] = has_hidden_images(text)
    features['has_suspicious_url_pattern'] = check_url_suspiciousness(text)
    features['has_high_html_ratio'] = calculate_html_text_ratio(text)
    features['has_common_typos'] = count_misspellings(text)
    features['has_spoofed_header'] = check_header_spoofing(text)
    features['is_newly_registered_domain'] = get_domain_age_days(text)
    features['has_typosquatting_link'] = check_for_typosquatting(text)
    
    # NEW ADVANCED FEATURES (4 more)
    features['has_suspicious_tld'] = check_domain_reputation(text)
    features['has_url_obfuscation'] = check_url_obfuscation(text)
    features['has_urgency_language'] = check_urgency_language(text)
    features['has_poor_grammar'] = check_grammar_quality(text)
    
    return features

if __name__ == "__main__":
    # Example usage for Typosquatting
    typosquatting_test = "URGENT: Click here: http://аррlе.com/login" # Cyrillic 'a' and 'p'
    print(f"Typosquatting Test: {get_advanced_features(typosquatting_test)['has_typosquatting_link']}")
