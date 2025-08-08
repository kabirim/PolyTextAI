import re
import html as ihtml
import string
import unicodedata
from dataclasses import dataclass
from typing import Literal, List, Tuple, Union, Optional, Dict

import langid
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

Lang = Literal["fr", "en"]

# -- Ressources NLTK : téléchargement silencieux si nécessaire
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        # Certaines versions exigent punkt_tab pour word_tokenize
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

_ensure_nltk()

@dataclass
class CleanOptions:
    """Options de nettoyage."""
    return_tokens: bool = False         # True -> liste de tokens, False -> string join
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    remove_digits: bool = False
    keep_hashtag_words: bool = True     # #mot -> "mot" si True, suppression totale si False
    remove_mentions: bool = True        # @pseudo
    min_token_len: int = 1              # supprime les tokens trop courts
    lang_threshold: float = 0.50        # seuil langid
    lowercase: bool = True
    strip_accents: bool = False         # True -> "é" -> "e"
    remove_emojis : bool = True

def remove_all_emojis(text: str) -> str:
    """
    Supprime tous les emojis Unicode (y compris smileys, drapeaux, mains, etc.)
    en se basant sur les plages Unicode connues.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symboles & pictogrammes
        "\U0001F680-\U0001F6FF"  # transport & cartes
        "\U0001F1E0-\U0001F1FF"  # drapeaux
        "\U00002700-\U000027BF"  # symboles divers
        "\U0001F900-\U0001F9FF"  # symboles supplémentaires
        "\U0001FA70-\U0001FAFF"  # symboles et pictogrammes étendus
        "\U00002600-\U000026FF"  # divers (soleil, météo, etc.)
        "\U00002B00-\U00002BFF"  # flèches et symboles divers
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)

def detect_lang_langid(text: str, threshold: float = 0.50) -> Lang:
    """Détecte 'fr' ou 'en' avec langid et lève une erreur sinon"""
    if not text or not text.strip():
        raise ValueError("Text is empty")
    lang, prob = langid.classify(text)
    if lang in ("fr", "en"):
        return lang  # type: ignore[return-value]
    raise ValueError(f"Language is not supported (langid: {lang}, p={prob:.2f}).")

def clean_text_from_html(text: str,
                         keep_hashtag_words: bool = True,
                         remove_mentions: bool = True) -> str:
    """Supprime balises HTML, emails, URLs, hashtags/mentions, normalise espaces."""
    if not text:
        return ""

    # Retirer balises HTML et décoder entités
    text = re.sub(r"<[^>]+>", " ", text)
    text = ihtml.unescape(text)

    # Emails
    text = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)

    # URLs
    url_pattern = (
        r"(?i)\b(?:https?://|ftp://|www\.)[^\s<>()]+"
        r"|(?:\b[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>()]*)?"
    )
    text = re.sub(url_pattern, " ", text)

    # Hashtags
    if keep_hashtag_words:
        # "#mot" -> " mot"
        text = re.sub(r"(?<!\w)#(\w+)", r" \1", text)
    else:
        text = re.sub(r"(?<!\w)#\w+", " ", text)

    # Mentions
    if remove_mentions:
        text = re.sub(r"(?<!\w)@\w+", " ", text)

    # Espaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    
def tokenize_text(text: str) -> List[str]:
        """Tokenisation simple via NLTK."""
        return word_tokenize(text)

def remove_punc(text: str) -> str:
    """Retire la ponctuation ASCII standard."""
    return text.translate(str.maketrans("", "", string.punctuation))

def _get_stopwords(lang: Lang) -> set:
    if lang == "en":
        return set(stopwords.words("english"))
    if lang == "fr":
        return set(stopwords.words("french"))
    return set()

def clean_raw_text(
    input_text: str,
    options: Optional[CleanOptions] = None
) -> Union[str, List[str], Dict[str, Union[str, List[str]]]]:
    """
    Pipeline de nettoyage : langue -> HTML/URLs/emails -> casse/accents -> ponctuation -> tokenisation -> stopwords.
    Retourne par défaut une chaîne nettoyée. Si options.return_tokens=True, retourne la liste de tokens.
    """
    if options is None:
        options = CleanOptions()

    if input_text is None:
        raise ValueError("input_text is None")

    text = input_text

    if options.lowercase:
        text = text.lower()

    # Détection de langue
    lang = detect_lang_langid(text, threshold=options.lang_threshold)
    
    # Suppression émojis
    if(options.remove_emojis):
        text = remove_all_emojis(text)
        
    # Nettoyage HTML/URLs/etc.
    text = clean_text_from_html(
        text,
        keep_hashtag_words=options.keep_hashtag_words,
        remove_mentions=options.remove_mentions,
    )

    # Accents
    if options.strip_accents:
        text = _strip_accents(text)

    # Ponctuation (avant tokenisation si on veut un texte plat)
    if options.remove_punctuation:
        text = remove_punc(text)

    # Chiffres
    if options.remove_digits:
        text = re.sub(r"\d+", " ", text)

    # Normalisation espaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenisation
    tokens = tokenize_text(text)

    # Stopwords + longueur minimale
    if options.remove_stopwords:
        sw = _get_stopwords(lang)
        tokens = [t for t in tokens if t not in sw]

    if options.min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= options.min_token_len]

    # Sortie
    if options.return_tokens:
        return tokens
    else:
        return " ".join(tokens)

# --- Exemples d'utilisation ---
if __name__ == "__main__":
    s = "Bonjour! <b>Ceci</b> est un test 😊. Site: https://exemple.fr #NLP @moi Email: a.b@example.com"
    print(clean_raw_text(s))  # string nettoyée par défaut
    print(clean_raw_text(s, CleanOptions(return_tokens=True, strip_accents=True)))