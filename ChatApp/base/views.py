from django.shortcuts import render
from django.http import JsonResponse
import random
import time
from agora_token_builder import RtcTokenBuilder
from .models import RoomMember
import json
from django.views.decorators.csrf import csrf_exempt

import cv2
import base64
import numpy as np
from django.http import JsonResponse

import mediapipe as mp

# Create your views here.

def lobby(request):
    return render(request, 'base/lobby.html')

def room(request):
    return render(request, 'base/room.html')


def getToken(request):
    appId = "1e03b384b24c46b897418c330b247eec"
    appCertificate = "4e44b60a67204d648bc3b299dc0d0354"
    channelName = request.GET.get('channel')
    uid = random.randint(1, 230)
    expirationTimeInSeconds = 3600
    currentTimeStamp = int(time.time())
    privilegeExpiredTs = currentTimeStamp + expirationTimeInSeconds
    role = 1

    token = RtcTokenBuilder.buildTokenWithUid(appId, appCertificate, channelName, uid, role, privilegeExpiredTs)

    return JsonResponse({'token': token, 'uid': uid}, safe=False)


@csrf_exempt
def createMember(request):
    data = json.loads(request.body)
    member, created = RoomMember.objects.get_or_create(
        name=data['name'],
        uid=data['UID'],
        room_name=data['room_name']
    )

    return JsonResponse({'name':data['name']}, safe=False)


def getMember(request):
    uid = request.GET.get('UID')
    room_name = request.GET.get('room_name')

    member = RoomMember.objects.get(
        uid=uid,
        room_name=room_name,
    )
    name = member.name
    return JsonResponse({'name':member.name}, safe=False)

@csrf_exempt
def deleteMember(request):
    data = json.loads(request.body)
    member = RoomMember.objects.get(
        name=data['name'],
        uid=data['UID'],
        room_name=data['room_name']
    )
    member.delete()
    return JsonResponse('Member deleted', safe=False)

def process_video(image_base64):
    # Base64 kodunu çöz
    image_bytes = base64.b64decode(image_base64)
    print("byte", image_bytes)
    # Bytes verisini numpy dizisine dönüştür
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    print("array", image_array)
    if image_array is not None:
        return"prdictions"
    else:
        return image_array

@csrf_exempt
def signLanguagePrediction(request):
   print("request:",request)
   if request.method == 'POST':
        print("gİRDİM")
        # Görüntüyü al
        data = json.loads(request.body)
        
        image_data = data.get('image')
        
        # Base64 kodunu çöz
        image_bytes = base64.b64decode(image_data)
        print("Selam")
        # Görüntüyü modele gönder ve sonucu al
        prediction = process_video(image_bytes)
        print("Data3", prediction)
        # Sonucu JSON formatında dön
        return JsonResponse({'prediction': prediction})

# def signLanguagePrediction(request):
#     return JsonResponse({'prediction':"ASLAPSA"})