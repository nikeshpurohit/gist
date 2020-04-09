from django.shortcuts import render
from django.http import HttpResponse
import gist_summarizer as gs

# Create your views here.
global context

def index(request, doc_type="none"):
    request.session['source_format'] = doc_type
    return render(request, 'gist/index.html')

def about(request):
    return render(request, 'gist/about.html')

def summary(request):
    doctype = str(request.GET.get('doc_type'))
    if doctype == "text":
        inText = []
        inText.append(str(request.GET.get('input-text')))
        cleanText = gs.preprocess_text(inText)
        summary = gs.summarise_from_clean_text(cleanText)
    elif doctype == "url":
        inText = "This is a url"
        cleanText = "This is a clean url"
        summary = "That was a url"
    context = {'inText': inText,
                'cleanText': cleanText,
                'summary': summary,
                'doc_type': doctype
                }
    return render(request, 'gist/summary.html', context)

def summarise(request):
    request.session['source_format'] = str(request.GET.get('doc_type'))
    if str(request.GET.get('doc_type')).lower() == "local":
        return index(request, "local")
    else:
        return render(request, 'gist/summarise.html')
