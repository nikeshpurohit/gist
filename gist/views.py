from django.shortcuts import render
from django.http import HttpResponse
import gist_abstractive as ga
import gist_extractive as ge
import gist_url2text as url2text
import gist_tools as tools
from textblob import TextBlob

# Create your views here.
global context


def index(request, doc_type="none"):
    request.session['source_format'] = doc_type
    return render(request, 'gist/index.html')


def about(request):
    return render(request, 'gist/about.html')


def summary(request):
    error = "none"
    doctype = str(request.POST.get('doc_type'))
    summary_method = request.POST['method']
    request.session['method'] = summary_method
    request.session['doc_type'] = doctype

    if summary_method == "abstractive":
        if doctype == "text":
            input = (str(request.POST.get('input-text')))
            request.session['input'] = input
            cleanText, summary = generate_abstractive(input)
        elif doctype == "url":
            input = url2text.get_content(str(request.POST.get('input-url')))
            request.session['input'] = input
            cleanText, summary = generate_abstractive(input)
        summary = ' '.join(summary)
        tempList = []
        for x in cleanText:
            tempList.append(x[0])
            cleanCleanText = tempList
        cleanCleanText = ' '.join(cleanCleanText)
        clean_length = tools.num_words(cleanCleanText)
        input_length = tools.num_words(input)
        summary_length = tools.num_words(summary)
        sum_ratio = round((summary_length / input_length), 3)
        reduction_percent = round((100 - (sum_ratio * 100)), 1)
        input_cleanwords = input_length - clean_length
    elif summary_method == "extractive":
        input = request.session['input']
        cleanText = [['some'], ['stuff']]
        summary = ge.summarize_text(input)
        input_length = tools.num_words(input)
        summary_length = tools.num_words(summary)
        clean_length = 0
        sum_ratio = round((summary_length / input_length), 3)
        reduction_percent = round((100 - (sum_ratio * 100)), 1)
        input_cleanwords = 0

    blob = TextBlob(input)
    detected_lang = blob.detect_language()
    if detected_lang != 'en':
        error = "not_en"

    context = {'input': input,#
               'cleanText': cleanText,#
               'summary': summary,#
               'doc_type': doctype,#
               'error': error,#
               'detected_lang': detected_lang,#
               'input_length': input_length,#
               'summary_length': summary_length,#
               'clean_length': clean_length,#
               'sum_ratio': sum_ratio,#
               'reduction_percent': reduction_percent,#
               'input_cleanwords': input_cleanwords,#
               'summary_method': summary_method,#
               }

    return render(request, 'gist/summary.html', context)


def summarise(request):
    request.session['source_format'] = str(request.POST.get('doc_type'))
    if str(request.POST.get('doc_type')).lower() == "local":
        return index(request, "local")
    else:
        return render(request, 'gist/summarise.html')


def generate_abstractive(input):
    cleanText = []
    summary = []
    sentenceSplit = ga.split_into_sentences(input)
    for i in sentenceSplit:
        cleanText.append(ga.preprocess_text(i))

    for j in cleanText:
        summary.append(ga.summarise_from_clean_text(j))
    return cleanText, summary
