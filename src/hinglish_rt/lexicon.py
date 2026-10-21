# Hinglish lexicon used by (a) the dataset generator and (b) the RL attacker's edit actions.
# Word-level tables + a loanword list. Expand with annotator feedback.
# NOTE: lexicon-based Hinglish is a controlled, rule-based approximation. It does not handle
# verb conjugation or Hindi word order. Verify a sample with bilingual annotators (see docs).

EN_TO_HI_ROMAN = {
    "ignore": ["ignore", "nazarandaz"],
    "disregard": ["nazarandaz", "ignore"],
    "forget": ["bhool", "bhool jao"],
    "previous": ["pehle wale", "pichle"],
    "prior": ["pehle ke", "pichle"],
    "earlier": ["pehle ke", "pichle"],
    "above": ["upar diye gaye", "upar wale"],
    "instructions": ["instructions", "nirdesh"],
    "instruction": ["instruction", "nirdesh"],
    "rules": ["niyam", "rules"],
    "rule": ["niyam", "rule"],
    "guidelines": ["guidelines", "dishanirdesh"],
    "restrictions": ["pabandiyan", "restrictions"],
    "reveal": ["reveal karo", "batao"],
    "show": ["dikhao", "dikha do"],
    "tell": ["batao", "bata do"],
    "print": ["print karo", "likh do"],
    "repeat": ["dohrao", "repeat karo"],
    "write": ["likho", "likh do"],
    "say": ["bolo", "keh do"],
    "answer": ["jawab", "answer"],
    "question": ["sawal", "question"],
    "secret": ["gupt", "secret"],
    "hidden": ["chhupe hue", "hidden"],
    "confidential": ["gopniya", "confidential"],
    "password": ["password", "pasword"],
    "everything": ["sab kuch", "har cheez"],
    "all": ["sabhi", "saare"],
    "nothing": ["kuch nahi"],
    "now": ["ab", "abhi"],
    "instead": ["iske bajay", "uske badle"],
    "only": ["sirf"],
    "never": ["kabhi nahi"],
    "always": ["hamesha"],
    "must": ["zaroor", "hona chahiye"],
    "please": ["please", "kripya"],
    "you": ["tum", "aap"],
    "your": ["tumhara", "aapka"],
    "i": ["main"],
    "my": ["mera", "meri"],
    "me": ["mujhe"],
    "we": ["hum"],
    "and": ["aur"],
    "or": ["ya"],
    "but": ["lekin", "par"],
    "because": ["kyunki"],
    "if": ["agar"],
    "then": ["toh"],
    "new": ["naye", "nayi"],
    "old": ["purane", "purani"],
    "follow": ["follow karo", "maano"],
    "obey": ["maano"],
    "stop": ["band karo"],
    "start": ["shuru karo"],
    "without": ["bina"],
    "exactly": ["bilkul"],
    "word": ["shabd"],
    "system": ["system", "tantra"],
    "the": [""],
    "don't": ["mat"],
    "dont": ["mat"],
}

EN_TO_HI_DEVA = {
    "ignore": ["इग्नोर"],
    "disregard": ["नज़रअंदाज़"],
    "forget": ["भूल"],
    "previous": ["पिछले"],
    "prior": ["पहले के"],
    "earlier": ["पहले के"],
    "above": ["ऊपर दिए गए"],
    "instructions": ["निर्देश"],
    "instruction": ["निर्देश"],
    "rules": ["नियम"],
    "rule": ["नियम"],
    "reveal": ["बताओ"],
    "show": ["दिखाओ"],
    "tell": ["बताओ"],
    "answer": ["जवाब"],
    "question": ["सवाल"],
    "secret": ["गुप्त"],
    "everything": ["सब कुछ"],
    "all": ["सभी"],
    "now": ["अब"],
    "instead": ["इसके बजाय"],
    "only": ["सिर्फ"],
    "never": ["कभी नहीं"],
    "always": ["हमेशा"],
    "you": ["तुम"],
    "your": ["तुम्हारे"],
    "i": ["मैं"],
    "my": ["मेरे"],
    "me": ["मुझे"],
    "and": ["और"],
    "or": ["या"],
    "but": ["लेकिन"],
    "new": ["नए"],
    "old": ["पुराने"],
    "without": ["बिना"],
    "the": [""],
}

FILLERS = ["yaar", "bhai", "na", "toh", "zara", "ek baar", "please", "jaldi"]

# English loanwords that appear inside Hinglish replacement strings (tagged as English by CMI).
_LOANWORDS = {
    "ignore", "reveal", "instructions", "instruction", "rules", "rule", "guidelines",
    "restrictions", "print", "repeat", "answer", "question", "secret", "hidden",
    "confidential", "password", "pasword", "please", "system", "prompt", "follow", "hidden",
    "word", "new",
}

# Curated Hindi-origin romanized words (language tagging for CMI).
_BASE_HI = {
    "pehle", "wale", "pichle", "purane", "nirdesh", "hidayatein", "bhool", "jao", "bhul",
    "sabhi", "saare", "niyam", "batao", "bata", "dikhao", "dikha", "karo", "ab", "abhi",
    "tum", "aap", "zaroor", "hona", "chahiye", "kripya", "aur", "iske", "bajay", "uske",
    "badle", "sirf", "maano", "naye", "nayi", "gupt", "nazarandaz", "ko", "ka", "ki", "ke",
    "se", "mein", "par", "hai", "hain", "nahi", "kya", "yeh", "woh", "mujhe", "mera", "meri",
    "kal", "banao", "kaise", "karein", "yaar", "bhai", "na", "toh", "zara", "ek", "baar",
    "jaldi", "padhai", "karte", "kaisa", "rahega", "liye", "tantra", "mat", "kabhi",
    "hamesha", "bina", "bilkul", "shabd", "jawab", "sawal", "dohrao", "likho", "likh",
    "bolo", "keh", "kuch", "har", "cheez", "upar", "diye", "gaye", "wala", "tumhara",
    "aapka", "main", "hum", "ya", "lekin", "kyunki", "agar", "purani", "band", "shuru",
    "pabandiyan", "dishanirdesh", "gopniya", "chhupe", "hue", "do", "dena", "kar",
}


def _derive_hi_words():
    words = set()
    for opts in EN_TO_HI_ROMAN.values():
        for opt in opts:
            for tok in opt.split():
                if tok and tok not in _LOANWORDS:
                    words.add(tok)
    return words


# Ambiguous with English words, so never tagged as Hindi on their own.
_AMBIGUOUS = {"do", "kar", "keh", "main", "par", "hum", "tantra"}
HI_ROMAN_WORDS = (_BASE_HI | _derive_hi_words()) - _AMBIGUOUS
