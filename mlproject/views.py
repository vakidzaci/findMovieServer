from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from nlp3 import findMovie
import simple as s
@csrf_exempt
def getMovies(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        return JsonResponse(findMovie(text),safe=False)
    if request.method == 'GET':
        return JsonResponse(findMovie("Sherlock"),safe=False)


@csrf_exempt
def getMovie(request):
    if request.method == 'GET':
        for i in request.META:
            print(i)
        # n  = int(request.META['n'])
        return JsonResponse({'id':10})
