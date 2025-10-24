from phonemizer import phonemize
words = ["angioplastie", "hémoglobine"]
for w in words:
    print(w, phonemize(w, language='fr-fr'))
