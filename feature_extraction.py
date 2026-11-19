
import re
import ipaddress
from urllib.parse import urlparse


def extract_features(url):
    """
    Extract numerical features from a URL.
    Does not visit or open the URL.
    """

    url = str(url).strip()

    # Handle URLs without an explicit scheme
    parsed_url = urlparse(
        url if "://" in url else "http://" + url
    )

    domain = parsed_url.hostname or ""

    # 1. URL and domain lengths
    url_length = len(url)
    domain_length = len(domain)

    # 2. Check whether the domain is an IP address
    is_domain_ip = 0
    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        pass

    # 3. Top-level domain
    tld = domain.split(".")[-1] if "." in domain else ""
    tld_length = len(tld)

    # 4. Number of subdomains
    domain_parts = domain.split(".") if domain else []
    no_of_subdomain = max(len(domain_parts) - 2, 0)

    # 5. Obfuscation indicators
    encoded_chars = re.findall(r"%[0-9a-fA-F]{2}", url)
    no_of_obfuscated_char = len(encoded_chars)
    has_obfuscation = int(no_of_obfuscated_char > 0)

    obfuscation_ratio = (
        no_of_obfuscated_char / url_length if url_length else 0
    )

    # 6. Letters and digits
    no_of_letters = sum(char.isalpha() for char in url)
    no_of_digits = sum(char.isdigit() for char in url)

    letter_ratio = no_of_letters / url_length if url_length else 0
    digit_ratio = no_of_digits / url_length if url_length else 0

    # 7. Special character counts
    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")

    other_special_chars = re.findall(
        r"[^A-Za-z0-9./:?=&%_-]",
        url
    )

    no_of_other_special_chars = len(other_special_chars)

    special_char_ratio = (
        no_of_other_special_chars / url_length
        if url_length else 0
    )

    # 8. HTTPS indicator
    is_https = int(parsed_url.scheme.lower() == "https")

    # Return features in a dictionary
    return {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated_char,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special_chars,
        "SpacialCharRatioInURL": special_char_ratio,
        "IsHTTPS": is_https
    }
