import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    """Remove filler words, stopwords, and lemmatize to minimize tokens."""
    doc = nlp(text)
    cleaned = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]
    return " ".join(cleaned)

def summarize_text(text: str, sentence_count: int = 3) -> str:
    """Summarize text to a fixed number of sentences to reduce length."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

def minimize_tokens(text: str) -> str:
    """Pipeline to clean + summarize text for minimal tokens."""
    cleaned = clean_text(text)
    minimized = summarize_text(cleaned, sentence_count=3)
    return minimized

if __name__ == "__main__":
    input_text = """
    ChatGPT is a large language model developed by OpenAI. It can answer questions, 
    generate code, assist with writing, and more. However, API usage costs depend 
    on the number of tokens processed. Reducing tokens is useful to minimize cost 
    while still preserving meaning.
    """
    print("Original:", input_text)
    print("\nMinimized:", minimize_tokens(input_text))
