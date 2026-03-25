import os
import re
import glob
from collections import Counter
from bs4 import BeautifulSoup
import PyPDF2

# text extraction

def extract_text_from_pdf(filepath):
    # read PDF pages
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
    return text

def extract_text_from_html(filepath):
    # parse HTML and clean
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        # remove tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'noscript', 'svg', 'meta', 'link']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        return text
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return ""


# preprocessing

def preprocess_text(text):
    # keep ASCII only
    text = text.encode('ascii', errors='ignore').decode('ascii')
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    # keep letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # normalize spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    # split + filter short tokens
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1 or t in ('a', 'i')]
    return tokens

# stopwords
STOPWORDS = {
    'the', 'is', 'in', 'it', 'of', 'and', 'to', 'a', 'for', 'on',
    'with', 'as', 'at', 'by', 'an', 'be', 'or', 'are', 'was', 'were',
    'this', 'that', 'from', 'has', 'have', 'had', 'will', 'can', 'do',
    'not', 'but', 'its', 'also', 'been', 'their', 'which', 'all',
    'more', 'other', 'would', 'there', 'than', 'may', 'should',
    'each', 'no', 'about', 'if', 'so', 'such', 'into', 'any',
    'he', 'she', 'they', 'we', 'you', 'who', 'what', 'when',
    'shall', 'will', 'may', 'must', 'could', 'would', 'should',
    'i', 'me', 'my', 'your', 'our', 'his', 'her', 'them', 'us',
    'these', 'those', 'then', 'where', 'how', 'why', 'up', 'out',
    'over', 'under', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'own', 'same', 'both', 'only', 'very',
    'just', 'being', 'having', 'doing', 'did', 'does', 'done',
    'ii', 'iii', 'iv', 'vi', 'vii', 'viii', 'ix', 'xi', 'xii',
    'per', 'etc', 'one', 'two', 'three', 'four', 'five',
}


if __name__ == '__main__':
    data_dir = 'e:/NLU2/data'

    # step 1: extract text
    print("="*50)
    print("STEP 1: TEXT EXTRACTION")
    print("="*50)

    documents = []  # (name, text)

    # PDFs
    pdf_dir = os.path.join(data_dir, 'pdf')
    pdf_files = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    print(f"\nFound {len(pdf_files)} PDF files:")
    for pf in pdf_files:
        fname = os.path.basename(pf)
        text = extract_text_from_pdf(pf)
        if text.strip():
            documents.append((fname, text))
            print(f"  {fname} -> {len(text)} chars")

    # HTML
    html_dir = os.path.join(data_dir, 'webpages')
    html_files = glob.glob(os.path.join(html_dir, '*.html')) + \
                 glob.glob(os.path.join(html_dir, '*.htm'))
    print(f"\nFound {len(html_files)} HTML files:")
    for hf in html_files:
        fname = os.path.basename(hf)
        text = extract_text_from_html(hf)
        if text.strip():
            documents.append((fname, text))
            print(f"  {fname} -> {len(text)} chars")

    print(f"\nTotal documents extracted: {len(documents)}")

    # save raw text
    with open('e:/NLU2/raw_corpus.txt', 'w', encoding='utf-8') as f:
        for name, text in documents:
            f.write(f"=== SOURCE: {name} ===\n")
            f.write(text + "\n\n")

    # step 2: preprocess
    print("\n" + "="*50)
    print("STEP 2: PREPROCESSING")
    print("="*50)

    cleaned_docs = []  # token lists
    all_tokens = []

    for name, text in documents:
        clean = preprocess_text(text)
        tokens = tokenize(clean)
        if len(tokens) > 10:  # skip short docs
            cleaned_docs.append(tokens)
            all_tokens.extend(tokens)
            print(f"  {name}: {len(tokens)} tokens")

    # save cleaned corpus
    # one doc per line
    with open('e:/NLU2/cleaned_corpus.txt', 'w', encoding='utf-8') as f:
        for tokens in cleaned_docs:
            f.write(' '.join(tokens) + '\n')
    print(f"\nCleaned corpus saved to cleaned_corpus.txt")

    # step 3: stats
    print("\n" + "="*50)
    print("STEP 3: DATASET STATISTICS")
    print("="*50)

    vocab = set(all_tokens)
    print(f"Total documents:  {len(cleaned_docs)}")
    print(f"Total tokens:     {len(all_tokens)}")
    print(f"Vocabulary size:  {len(vocab)}")
    print(f"Avg tokens/doc:   {len(all_tokens) / len(cleaned_docs):.1f}")

    # freq (no stopwords)
    tokens_no_stop = [t for t in all_tokens if t not in STOPWORDS]
    word_freq = Counter(tokens_no_stop)

    print(f"\nTop 25 most frequent words (excluding stopwords):")
    for word, count in word_freq.most_common(25):
        print(f"  {word}: {count}")

    # step 4: word cloud
    print("\n" + "="*50)
    print("STEP 4: WORD CLOUD")
    print("="*50)

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wc_text = ' '.join(tokens_no_stop)
    wc = WordCloud(width=1200, height=600, background_color='white',
                  max_words=150, colormap='viridis',
                  min_font_size=8)
    wc.generate(wc_text)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - IIT Jodhpur Corpus', fontsize=16)
    plt.tight_layout()
    plt.savefig('e:/NLU2/wordcloud.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Word cloud saved to wordcloud.png")

    # save stats
    with open('e:/NLU2/dataset_stats.txt', 'w') as f:
        f.write("DATASET STATISTICS - IIT JODHPUR CORPUS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Sources used:\n")
        for name, _ in documents:
            f.write(f"  - {name}\n")
        f.write(f"\nTotal documents:  {len(cleaned_docs)}\n")
        f.write(f"Total tokens:     {len(all_tokens)}\n")
        f.write(f"Vocabulary size:  {len(vocab)}\n")
        f.write(f"Avg tokens/doc:   {len(all_tokens)/len(cleaned_docs):.1f}\n\n")
        f.write("Top 30 most frequent words (excluding stopwords):\n")
        for word, count in word_freq.most_common(30):
            f.write(f"  {word}: {count}\n")
    print("Stats saved to dataset_stats.txt")