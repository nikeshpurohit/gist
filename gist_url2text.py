from goose3 import Goose
g = Goose()

def get_title(url):
    article = g.extract(url=url)
    return article.title

def get_content(url):
    article = g.extract(url=url)
    return article.cleaned_text
