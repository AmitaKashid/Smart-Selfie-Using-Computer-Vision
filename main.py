import os
import cv2
import urllib
import numpy as np
from werkzeug.utils import secure_filename
from urllib.request import Request, urlopen
from flask import Flask, render_template, Response, request, redirect, flash, url_for


from camera import VideoCamera
from Graphical_Visualisation import Emotion_Analysis


app = Flask(__name__)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def gen(camera):
    "" "Helps in Passing frames from Web Camera to server"""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')




@app.route('/')
def Start():
    return render_template('Start.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/RealTime', methods=['POST'])
def RealTime():    
    return render_template('RealTime.html')


@app.route('/takeimage', methods=['POST'])
def takeimage():
    """ Captures Images from WebCam, saves them, does Emotion Analysis"""

    v = VideoCamera()
    _, frame = v.video.read()
    

    result = Emotion_Analysis("capture.jpg")

    return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2], music=result[3],
                           sentence='happy', activity='happy', image=result[3], link='happy')


if __name__ == '__main__':
    app.run(debug=True)
