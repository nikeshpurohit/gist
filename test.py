import gist_abstractive as gs
import gist_url2text as url2text

input = 'this is one sentent. this is tww. three.'

#cleanText = gs.preprocess_text(input)
#summary = gs.summarise_from_clean_text(cleanText)
cleanText = []
summary = []
sentenceSplit = gs.split_into_sentences(input)
for i in sentenceSplit:
    cleanText.append(gs.preprocess_text(i))

for j in cleanText:
    summary.append(gs.summarise_from_clean_text(j))

print("orig:", sentenceSplit)
print("clean:", cleanText)
print("summ:", summary)
