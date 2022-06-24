import imp
from django.shortcuts import render
from django.http import HttpResponse
from ml_utils.utils import pre_process
# Create your views here.

def index(request):
    context={
        'a':'himanshu yadav will deploy model in less than two hours'
    }
    return render(request,'index.HTML',context)

def predict(request):
    
    if request.method=='POST':
        context={
            'status':pre_process(request.POST.dict())
        }
    
    return render(request,'index.HTML', context)
