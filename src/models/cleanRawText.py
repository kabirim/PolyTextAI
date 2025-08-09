import re
import html as ihtml
import string
import unicodedata
from dataclasses import dataclass
from typing import Literal, List, Union, Optional, Dict, Set

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
    """Options de nettoyage adaptées aux CV."""
    # Sortie
    return_tokens: bool = False          # True -> liste de tokens, False -> string join
    return_meta: bool = False            # True -> dict {"lang","text","tokens"}

    # Comportement de base (léger par défaut)
    lowercase: bool = False              # ⚠️ garder la casse pour technos
    strip_accents: bool = False
    remove_emojis: bool = True

    # HTML/whitespace
    keep_hashtag_words: bool = True
    remove_mentions: bool = True
    preserve_linebreaks: bool = True     # garder la structure (lignes/puces)
    collapse_blank_lines: bool = True    # compacter \n\n\n -> \n\n

    # Ponc./digits/stopwords
    remove_punctuation: bool = False     # ⚠️ par défaut on garde la ponctuation
    punctuation_keep: Set[str] = None    # si on retire la ponct., ces symboles sont préservés
    remove_digits: bool = False
    remove_stopwords: bool = False       # ⚠️ éviter de supprimer "R", "C", etc.
    min_token_len: int = 1

    # Retour texte nettoyé (non tokenisé)
    return_cleaned_text: bool = True

    # Langue
    lang_threshold: float = 0.50

    def __post_init__(self):
        if self.punctuation_keep is None:
            # Symboles fréquents dans les technos
            self.punctuation_keep = {"+", "#", ".", "-", "_", "/", "&"}

def remove_all_emojis(text: str) -> str:
    """Supprime tous les emojis Unicode."""
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symboles & pictogrammes
        "\U0001F680-\U0001F6FF"  # transport & cartes
        "\U0001F1E0-\U0001F1FF"  # drapeaux
        "\U00002700-\U000027BF"  # symboles divers
        "\U0001F900-\U0001F9FF"  # symboles supplémentaires
        "\U0001FA70-\U0001FAFF"  # pictos étendus
        "\U00002600-\U000026FF"  # météo etc.
        "\U00002B00-\U00002BFF"  # flèches/symboles
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)

def detect_lang_langid(text: str, threshold: float = 0.50) -> Lang:
    """Détecte 'fr' ou 'en' avec langid et lève une erreur sinon (respecte le seuil)."""
    if not text or not text.strip():
        raise ValueError("Text is empty")
    lang, prob = langid.classify(text)
    if lang in ("fr", "en"):
        return lang  # type: ignore[return-value]
    raise ValueError(f"Language is not supported (langid: {lang}, p={prob:.2f}).")

def clean_text_from_html(
    text: str,
    keep_hashtag_words: bool = True,
    remove_mentions: bool = True,
    preserve_linebreaks: bool = True,
) -> str:
    """Supprime balises HTML, emails, URLs, hashtags/mentions. Gère les sauts de ligne."""
    if not text:
        return ""

    # Si HTML probable, convertit certains tags en \n pour préserver la structure
    if preserve_linebreaks:
        # Insère des \n avant de supprimer le reste des balises
        text = re.sub(r"(?i)</?(br|p|div|li|ul|ol|h[1-6]|section|article|hr)[^>]*>", "\n", text)
    
    # Supprime le reste des balises
    text = re.sub(r"<[^>]+>", " ", text)

    # Décoder les entités HTML
    text = ihtml.unescape(text)

    # Emails
    text = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)

    # URLs (http/https/ftp/www + domaines nus)
    url_pattern = (
        r"(?i)\b(?:https?://|ftp://|www\.)[^\s<>()]+"
        r"|(?:\b[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>()]*)?"
    )
    text = re.sub(url_pattern, " ", text)

    # Hashtags
    if keep_hashtag_words:
        text = re.sub(r"(?<!\w)#(\w+)", r" \1", text)
    else:
        text = re.sub(r"(?<!\w)#\w+", " ", text)

    # Mentions
    if remove_mentions:
        text = re.sub(r"(?<!\w)@\w+", " ", text)

    # Normalisation espaces SANS écraser les \n si demandé
    if preserve_linebreaks:
        # condense espaces sur chaque ligne, mais garde les retours
        text = re.sub(r"[ \t\f\v]+", " ", text)
        # normalise les fins de ligne (Windows/Mac → \n)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # compacte lignes vides excessives
        text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
        text = text.strip()
    else:
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

def preserve_tokenize_text(text: str) -> List[str]:
    """
    Tokenisation qui préserve les symboles fréquents dans les noms de technologies :
    C#, C++, .NET, Node.js, ASP.NET, Docker/Kubernetes, etc.
    """
    pattern = r"[A-Za-z0-9]+(?:[+#./_-][A-Za-z0-9]+)*|[+#./_-][A-Za-z0-9]+"
    return re.findall(pattern, text)

def tokenize_text_preserve_symbols(text: str) -> List[str]:
    """
    Tokenisation qui préserve les symboles fréquents dans les noms de technologies :
    C#, C++, .NET, Node.js, ASP.NET, Docker/Kubernetes, etc.
    """
    pattern = re.compile(r"""
        # Mot alphanumérique, éventuellement relié par . / _ -, puis
        #    éventuellement terminé par un ou plusieurs # ou +
        (?:[A-Za-z0-9]+(?:[./_-][A-Za-z0-9]+)*(?:[#+]+)?) 
        |
        # Mot qui COMMENCE par un symbole (. / _ -), puis alphanumériques,
        #    éventuellement reliés, puis suffixe #/+ optionnel
        (?:[.#/_-][A-Za-z0-9]+(?:[./_-][A-Za-z0-9]+)*(?:[#+]+)?) 
    """, re.VERBOSE)
    return pattern.findall(text)

def remove_punc_keep(text: str, keep: Set[str]) -> str:
    """
    Retire la ponctuation ASCII standard en préservant certains symboles (ex: + # . - _ / &).
    """
    # Construire la table de traduction en retirant seulement ce qu'on NE garde pas
    to_remove = set(string.punctuation) - set(keep)
    table = str.maketrans("", "", "".join(sorted(to_remove)))
    return text.translate(table)

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
    Pipeline "light clean" pour CV :
    - Garde casse et symboles techniques.
    - Préserve la structure (lignes/puces).
    - Optionnel : suppression emojis, accents, stopwords, ponctuation (avec liste de symboles préservés).
    - Sortie : texte nettoyé, tokens, ou méta.
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

    # Emojis
    if options.remove_emojis:
        text = remove_all_emojis(text)

    # HTML/URLs/etc.
    text = clean_text_from_html(
        text,
        keep_hashtag_words=options.keep_hashtag_words,
        remove_mentions=options.remove_mentions,
        preserve_linebreaks=options.preserve_linebreaks,
    )

    # Accents
    if options.strip_accents:
        text = _strip_accents(text)

    # Ponctuation (en préservant les symboles techniques)
    if options.remove_punctuation:
        text = remove_punc_keep(text, keep=options.punctuation_keep)

    # Chiffres
    if options.remove_digits:
        text = re.sub(r"\d+", " ", text)

    # Normalisation espaces finale
    if options.preserve_linebreaks:
        text = re.sub(r"[ \t]+", " ", text)
        if options.collapse_blank_lines:
            text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
    else:
        text = re.sub(r"\s+", " ", text).strip()

    # Tokenisation (préserve C#, C++, .NET, Node.js, ...)
    tokens = tokenize_text_preserve_symbols(text)

    # Stopwords + longueur minimale
    if options.remove_stopwords:
        sw = _get_stopwords(lang)
        tokens = [t for t in tokens if t.lower() not in sw]

    if options.min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= options.min_token_len]

    # Sorties (ordre de priorité)
    if options.return_meta:
        return {"lang": lang, "text": None if options.return_tokens else " ".join(tokens), "tokens": tokens}
    if options.return_tokens:
        return tokens
    if options.return_cleaned_text:
        return text
    return " ".join(tokens)

# --- Exemples d'utilisation ---
if __name__ == "__main__":
    s = """Bonjour! <b>Ceci</b> est un test 😊.
    Compétences: C++, C#, .NET, React.js, Node.js, Docker/Kubernetes
    Site: https://exemple.fr #NLP @moi Email: a.b@example.com
    - Expérience: DevOps - AWS & Terraform
    """
    print(clean_raw_text(s))  # string nettoyée par défaut (light clean)
    print(clean_raw_text(s, CleanOptions(return_tokens=True, strip_accents=True)))
    print(clean_raw_text(s, CleanOptions(remove_punctuation=True, return_tokens=False)))