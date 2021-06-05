from django.http import HttpResponse
from django.shortcuts import render
# from mining.entry import mining
# from mining.whole_model import mine_model
from only_web. mining.whole_model import decode
from django.views.decorators.csrf import csrf_exempt
import json

def index(request):
    return render(request, "index.html", {})


def mine(request):
    return HttpResponse(decode('input.txt'))


def sample(request):
    return HttpResponse(decode('input.txt'))


def inference(request):
    try:
        # mine_model('input.txt')
        print('')
    except Exception:
        return HttpResponse(json.dumps({'result': Exception}))
    else:
        return HttpResponse(json.dumps({'result': 'success'}))


@csrf_exempt
def upload(request):
    if request.method == 'POST':
        files = request.FILES.getlist('fileToUpload', None)
        path = 'input.txt'
        with open(path, 'w', encoding='utf8') as f:
            for file in files:
                for line in file.readlines():
                    a = bytes.decode(line)
                    f.write(a.replace('\r', ''))
    return HttpResponse(json.dumps({'result': 'success'}))