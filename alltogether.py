import kivy
from detector.face_detector import MTCNNFaceDetector
from models.elg_keras import KerasELG
from keras import backend as K

import math
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from matplotlib import pyplot as plt

from kivy.clock import Clock

from kivy.uix.progressbar import ProgressBar
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.widget import Widget

import results as r

import json
data = {}
data['dist'] = []  

def makeanim(pupil_center_left_cp, pupil_center_right_cp, path):
    width = pupil_center_right_cp.shape[1]
    height = pupil_center_right_cp.shape[0]
    FPS = 1
    seconds = 2

    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(path, fourcc, float(FPS), (width, height))

    a=0
    for _ in range(FPS*seconds):
        if a%2==0:
            video.write(pupil_center_left_cp)
        else:
            video.write(pupil_center_right_cp)
        a+=1
    video.release()
    
#########################################################################################
"""Progress BAr Animation"""
class ProgressAnimator(object):
    """
    This class handles the animation of a 'ProgressBar' instance given a list
    of functions that do the real work. Each of these functions need to accept
    a single 'callback' parameter specifying the function to call when done.
    """
    def __init__(self, progressbar, callbacks):
        self.pb = progressbar  # Reference to the progress bar  to update
        self.pb.max, self.pb.value = len(callbacks), 0
        self.callbacks = callbacks  # A list of callback functions doing work
        self.index = 0  # The index of the callback being executed
        Clock.schedule_once(lambda dt: callbacks[0](self.task_complete), 0.1)

    def task_complete(self):
        """ The last called heavy worker is done. See if we have any left """
        self.index += 1
        self.pb.value = self.index
        if self.index < len(self.callbacks):
            Clock.schedule_once(
                lambda dt: self.callbacks[self.index](self.task_complete), 0.1)

##########################################################################################

class MyGrid(GridLayout):
    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)
        self.cols = 2

        self.inside = GridLayout()
        self.inside.cols = 1

        self.inside.add_widget(Label(text="Name: "))
        self.name = TextInput(multiline=False)
        self.inside.add_widget(self.name)

        self.inside.add_widget(Label(text="Distance for each eye: "))
        self.lastName = TextInput(multiline=False)
        self.inside.add_widget(self.lastName)

        self.inside.add_widget(Label(text="Distance for Both eyes: "))
        self.email = TextInput(multiline=False)
        self.inside.add_widget(self.email)

        self.inside.add_widget(Label(text="Max X Resolution: "))
        self.xres = TextInput(multiline=False)
        self.inside.add_widget(self.xres)
        
        self.name.text = ""
        self.lastName.text = "50"
        self.email.text = "1000000"
        self.xres.text = "1280"

        self.pb = ProgressBar()
        self.inside.add_widget(self.pb)

        self.inside.add_widget(Label(text=" "))
        
        self.submit = Button(text="RUN FOR RIGHT EYE", font_size=30)
        self.submit.bind(on_press=self.pressedrunright)
        self.inside.add_widget(self.submit)

        self.submit = Button(text="RUN FOR LEFT EYE", font_size=30)
        self.submit.bind(on_press=self.pressedrunleft)
        self.inside.add_widget(self.submit)

        self.submit = Button(text="RUN FOR BOTH EYE", font_size=30)
        self.submit.bind(on_press=self.pressedrune)
        self.inside.add_widget(self.submit)
        
        self.inside.add_widget(Label(text=" "))

        self.submit = Button(text="SHOW RESULTS", font_size=20)
        self.submit.bind(on_press=self.pressresults)
        self.inside.add_widget(self.submit)

        
        self.add_widget(self.inside)
##########################################################
        self.inside2 = GridLayout()
        self.inside2.cols = 3

        self.submit = Button(text="Right eye: --<.>--", font_size=20)
        self.submit.bind(on_press=self.pressedrtw)
        self.inside2.add_widget(self.submit)
        
        self.submit = Button(text="Right eye: ----->", font_size=20)
        self.submit.bind(on_press=self.pressedrtq)
        self.inside2.add_widget(self.submit)

        self.submit = Button(text="Right eye: <-----", font_size=20)
        self.submit.bind(on_press=self.pressedrte)
        self.inside2.add_widget(self.submit)

        self.image1 = Image(source='./one/right/rightmid.jpg')
        self.image1.allow_stretch= True
        #self.image.height=5000
        self.inside2.add_widget(self.image1)

        self.image2 = Image(source='./one/right/right1.jpg')
        self.image2.allow_stretch= True
        self.inside2.add_widget(self.image2)

        self.image3 = Image(source='./one/right/right2.jpg')
        self.image3.allow_stretch= True
        self.inside2.add_widget(self.image3)

        self.inside2.add_widget(Label(text=" "))
        self.inside2.add_widget(Label(text=" "))
        self.inside2.add_widget(Label(text=" "))

        Clock.schedule_interval(self.update_pic,1)

        self.submit = Button(text="Left eye: --<.>--", font_size=20)
        self.submit.bind(on_press=self.pressedlfw)
        self.inside2.add_widget(self.submit)
        
        self.submit = Button(text="Left eye: ----->", font_size=20)
        self.submit.bind(on_press=self.pressedlfq)
        self.inside2.add_widget(self.submit)

        self.submit = Button(text="Left eye: <-----", font_size=20)
        self.submit.bind(on_press=self.pressedlfe)
        self.inside2.add_widget(self.submit)

        self.image4 = Image(source='./one/left/leftmid.jpg')
        self.inside2.add_widget(self.image4)

        self.image5 = Image(source='./one/left/left1.jpg')
        self.inside2.add_widget(self.image5)

        self.image6 = Image(source='./one/left/left2.jpg')
        self.inside2.add_widget(self.image6)

        self.inside2.add_widget(Label(text=" "))
        self.inside2.add_widget(Label(text=" "))
        self.inside2.add_widget(Label(text=" "))

        Clock.schedule_interval(self.update_pic,1)

        self.submit = Button(text="Both eyes: --<.>--", font_size=20)
        self.submit.bind(on_press=self.pressedbmd)
        self.inside2.add_widget(self.submit)
        
        self.submit = Button(text="Both eye: ----->", font_size=20)
        self.submit.bind(on_press=self.pressedbrt)
        self.inside2.add_widget(self.submit)

        self.submit = Button(text="Both eye: <-----", font_size=20)
        self.submit.bind(on_press=self.pressedblf)
        self.inside2.add_widget(self.submit)

        self.image7 = Image(source='./both/bmd.jpg')
##        self.image.size_hint_y= None
##        self.image.size_hint_x= None
##        self.image.width=100
##        self.image.height=50
##        self.image.allow_stretch= True
##        #self.image.height=5000
        self.inside2.add_widget(self.image7)

        self.image8 = Image(source='./both/brt.jpg')
        self.inside2.add_widget(self.image8)

        self.image9 = Image(source='./both/blf.jpg')
        self.inside2.add_widget(self.image9)

        Clock.schedule_interval(self.update_pic,1)
        
        self.add_widget(self.inside2)   

    def update_pic(self,dt):
        self.image1.reload()
        self.image2.reload()
        self.image3.reload()
        self.image4.reload()
        self.image5.reload()
        self.image6.reload()
        self.image7.reload()
        self.image8.reload()
        self.image9.reload()

    def pressresults(self,instance):
        Eye_Squint_Detector().stop() 
        r.Results().run()

        
    def pressedrune(self, instance):
        self.pb.value=10
        
        name = self.name.text
        dist_one = self.lastName.text

        print("Name:", name)

        """## Instantiate GazeML ELG model

        ##A stacked hourglass framework for iris and eye-lid detection.
        """

        model = KerasELG()
        model.net.load_weights("./elg_weights/elg_keras.h5")
        
        self.pb.value=10
        
        cnt_both=0
        
        while(cnt_both<3):

            if cnt_both == 0:
                fn1 = "./both/bmd.jpg"
                input_img = cv2.imread(fn1)[..., ::-1]

            if cnt_both == 1:
                fn2 = "./both/brt.jpg"
                input_img = cv2.imread(fn2)[..., ::-1]

            if cnt_both == 2:
                fn3 = "./both/blf.jpg"
                input_img = cv2.imread(fn3)[..., ::-1]
                
            self.pb.value=20
            
            im_dim=input_img.shape
            print("dimesnsions of image are:")
            print("X=")
            print(im_dim[1])
            print("Y=")
            print(im_dim[0])

            plt.title('Input image')
            plt.imshow(input_img)
            plt.show()

            #########################################################
            #cutting in half
            left_eye_im = input_img[
                int(0):int(im_dim[0]),
                int(im_dim[1]/2):int(im_dim[1]), :]
            #left_eye_im = left_eye_im[:,::-1,:] # No need for flipping left eye for iris detection
            right_eye_im = input_img[
                int(0):int(im_dim[0]),
                int(0):int(im_dim[1]/2), :]
            self.pb.value=30
            ##########################################################
            
            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            plt.title('Left eye')
            plt.imshow(left_eye_im)
            plt.subplot(1,2,2)
            plt.title('Right eye')
            plt.imshow(right_eye_im)
            plt.show()

            dim=left_eye_im.shape
            print("dimensions are") 
            print(dim[0])
            print(dim[1])

            """## Preprocess eye images

            ELG has fixed input shape of (108, 180, 1).

            Input images are first converted to grey scale, thrown into the histogram equalization process, and finally rescaled to [-1, +1].
            """

            inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
            inp_left = cv2.equalizeHist(inp_left)
            inp_left = cv2.resize(inp_left, (180,108))[np.newaxis, ..., np.newaxis]

            inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
            inp_right = cv2.equalizeHist(inp_right)
            inp_right = cv2.resize(inp_right, (180,108))[np.newaxis, ..., np.newaxis]
            self.pb.value=50
            
            plt.figure(figsize=(8,3))
            plt.subplot(1,2,1)
            plt.title('Left eye')
            plt.imshow(inp_left[0,...,0], cmap="gray")
            plt.subplot(1,2,2)
            plt.title('Right eye')
            plt.imshow(inp_right[0,...,0], cmap="gray")
            plt.show()

            """## Predict eye region landmarks

            ELG forwardpass. Output shape: (36, 60, 18)
            """

            inp_left.shape, inp_right.shape

            input_array = np.concatenate([inp_left, inp_right], axis=0)
            pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

            """## Visualize output heatmaps

            Eighteen heatmaps are predicted: 
            - Eight heatmaps for iris (green)
            - Eight heatmaps for eye-lid (red)
            - Two heatmaps for pupil (blue)
            """

            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.axis('off')
            plt.title('Left eye heatmaps')
            hm_r = np.max(pred_left[...,:8], axis=-1, keepdims=True)
            hm_g = np.max(pred_left[...,8:16], axis=-1, keepdims=True)
            hm_b = np.max(pred_left[...,16:], axis=-1, keepdims=True)
            plt.imshow(np.concatenate([hm_r, hm_g, hm_b], axis=-1))
            plt.subplot(1,2,2)
            plt.axis('off')
            plt.title('Right eye heatmaps')
            hm_r = np.max(pred_right[...,:8], axis=-1, keepdims=True)
            hm_g = np.max(pred_right[...,8:16], axis=-1, keepdims=True)
            hm_b = np.max(pred_right[...,16:], axis=-1, keepdims=True)
            plt.imshow(np.concatenate([hm_r, hm_g, hm_b], axis=-1))
            plt.show()
            self.pb.value=70

            """# Draw eye region landmarks"""
            pnt_center=[]

            def draw_pupil(im, inp_im, lms):
                draw = im.copy()
                draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
                pupil_center = np.zeros((2,))
                pnts_outerline = []
                pnts_innerline = []
                stroke = inp_im.shape[1] // 12 + 1
                for i, lm in enumerate(np.squeeze(lms)):
                    #print(lm)
                    y, x = int(lm[0]*3), int(lm[1]*3)

                    if i < 8:
                        #draw = cv2.circle(draw, (y, x), stroke, (125,255,125), -1)
                        pnts_outerline.append([y, x])
                    elif i < 16:
                        #draw = cv2.circle(draw, (y, x), stroke, (125,125,255), -1)
                        pnts_innerline.append([y, x])
                        pupil_center += (y,x)
                #     elif i < 17:
                #         draw = cv2.drawMarker(draw, (y, x), (255,200,200), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke, line_type=cv2.LINE_AA)
                #     else:
                #         draw = cv2.drawMarker(draw, (y, x), (255,125,125), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke, line_type=cv2.LINE_AA)
                pupil_center = (pupil_center/8).astype(np.int32)
                draw = cv2.cv2.circle(draw, (pupil_center[0], pupil_center[1]), stroke//6, (255,255,0), -1)        
                draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(125,255,125), thickness=stroke//5)
                draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(125,125,255), thickness=stroke//5)
                
                pnt_center.append(pupil_center[0])
                print(pupil_center[0])
                
                
                pnt_center.append(pupil_center[1])
                print(pupil_center[1])
                

                return draw

            
            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            plt.title("Left eye")
            lms_left = model._calculate_landmarks(pred_left)
            result_left = draw_pupil(left_eye_im, inp_left, lms_left)
            plt.imshow(result_left)
            plt.subplot(1,2,2)
            plt.title("Right eye")
            lms_right = model._calculate_landmarks(pred_right)
            result_right = draw_pupil(right_eye_im, inp_right, lms_right)
            plt.imshow(result_right)
            plt.show()
            
            self.pb.value=80
            
            draw2 = input_img.copy()
            dim2=input_img.shape


            a=int(pnt_center[1]*(dim[0]/108)) 
            b=int(pnt_center[0]*(dim[1]/180) + int(im_dim[1]/2))
            c=int(pnt_center[3]*(dim[0]/108)) 
            d=int(pnt_center[2]*(dim[1]/180)) 

            print("left_eye Y")
            print(a)
            print("left_eye X")
            print(b)

            print("right_eye Y")
            print(c)
            print("right_eye X")
            print(d)

            slice_h = slice(int(0), int(im_dim[0]))
            slice_w = slice(int(im_dim[1]/2), int(im_dim[1]))
            im_shape = left_eye_im.shape[::-1]

            draw2[slice_h, slice_w, :] = cv2.resize(result_left, im_shape[1:])

            slice_h = slice(int(0), int(im_dim[0]))
            slice_w = slice(int(0), int(im_dim[1]/2))
            im_shape = right_eye_im.shape[::-1]

            net_dist=math.sqrt((a-c)*(a-c)+(b-d)*(b-d))
            draw2[slice_h, slice_w, :] = cv2.resize(result_right, im_shape[1:])
            plt.imshow(draw2)
            plt.show()

            font = cv2.FONT_HERSHEY_SIMPLEX
            self.pb.value=100
            print("The net Distance between the pupil Centres is :")
            print(net_dist)
        
            measured=int(self.lastName.text)
            max_x_res=int(self.xres.text)
            print(round(net_dist*(measured/max_x_res),2))


            draw3=cv2.line(draw2, (b, a), (d, c), (255, 0, 0), 2)
            dim3=draw2.shape

            # print("input_dim")
            # print(dim2)
            # print("output_dim")
            # print(dim3)
            
            pnt_center=[]
            
            #cv2.putText(draw3,net_dist,(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
            plt.figure(figsize=(10,10))
            plt.axis('off')
            plt.imshow(draw2)
            plt.show()
            plt.imshow(draw3)
            plt.show()
            
            cv2.waitKey(1000)
            #cv2.destroyAllWindows()
            self.pb.value=0

            
##            data['dist'] = []
##            if cnt_both==0:
##                a=str(round(net_dist*(measured/max_x_res),2))
##                print(a)
##            if cnt_both==1:
##                b=str(round(net_dist*(measured/max_x_res),2))
##            if cnt_both==2:
##                c=str(round(net_dist*(measured/max_x_res),2))
##                data['dist'].append({  
##                '1': a,
##                '2': b,
##                '3': c
##                })
##            with open('data.txt', 'w') as outfile:  
##                json.dump(data, outfile)
            cnt_both+=1
        
        
    def pressedrunright(self, instance):
        name = self.name.text
        dist_both = self.email.text

        print("Name:", name)


        """## Instantiate GazeML ELG model

        #A stacked hourglass framework for iris and eye-lid detection.
        """

        model = KerasELG()
        model.net.load_weights("./elg_weights/elg_keras.h5")

        cnt_one=0
        while cnt_one<3:
            
            if cnt_one==0:
                fnr = "./one/right/right1.jpg"
                right_eye_im = cv2.imread(fnr)[..., ::-1]
                dim_x1=right_eye_im.shape[1]
                dim_y1=right_eye_im.shape[0]
                
                fnl = "./one/right/right2.jpg"
                left_eye_im = cv2.imread(fnl)[..., ::-1]
                dim_x2=left_eye_im.shape[1]
                dim_y2=left_eye_im.shape[0]

                img_name_lf = "./output/right/left.jpg"
                img_name_rt = "./output/right/rt.jpg"

                makeanim(right_eye_im,left_eye_im,'./anim/one/right_lf_rt')
            
            if cnt_one==1:
                fnr = "./one/right/right2.jpg"
                right_eye_im = cv2.imread(fnr)[..., ::-1]
                dim_x1=right_eye_im.shape[1]
                dim_y1=right_eye_im.shape[0]
                
                fnl = "./one/right/rightmid.jpg"
                left_eye_im = cv2.imread(fnl)[..., ::-1]
                dim_x2=left_eye_im.shape[1]
                dim_y2=left_eye_im.shape[0]
                
                img_name_md = "./output/right/mid.jpg"

                makeanim(right_eye_im,left_eye_im,'./anim/one/right_lf_md')
                

                img_name = "./output/right/rightmid2.jpg"
            if cnt_one==2:
                fnr = "./one/right/right1.jpg"
                right_eye_im = cv2.imread(fnr)[..., ::-1]
                dim_x1=right_eye_im.shape[1]
                dim_y1=right_eye_im.shape[0]
                
                fnl = "./one/right/rightmid.jpg"
                left_eye_im = cv2.imread(fnl)[..., ::-1]
                dim_x2=left_eye_im.shape[1]
                dim_y2=left_eye_im.shape[0]
                img_name = "./output/right/rightmid1.jpg"

                makeanim(right_eye_im,left_eye_im,'./anim/one/right_rt_md')
                
            pupil_center_right_cp=right_eye_im.copy()
            pupil_center_left_cp=left_eye_im.copy()

            #########################################################
            width = pupil_center_right_cp.shape[1]
            height = pupil_center_right_cp.shape[0]
            FPS = 1
            seconds = 2

            fourcc = VideoWriter_fourcc(*'MP42')
            video = VideoWriter('./circle_noise.avi', fourcc, float(FPS), (width, height))

            a=0
            for _ in range(FPS*seconds):
                if a%2==0:
                    video.write(pupil_center_left_cp)
                else:
                    video.write(pupil_center_right_cp)
                a+=1
            video.release()
            #########################################################


            plt.subplot(1,2,1)
            plt.title('Input image rt')
            plt.imshow(right_eye_im)
            plt.subplot(1,2,2)
            plt.title('Input image lf')
            plt.imshow(left_eye_im)
            plt.show()
            
            inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
            inp_left = cv2.equalizeHist(inp_left)
            inp_left = cv2.resize(inp_left, (180,108))[np.newaxis, ..., np.newaxis]

            inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
            inp_right = cv2.equalizeHist(inp_right)
            inp_right = cv2.resize(inp_right, (180,108))[np.newaxis, ..., np.newaxis]

            plt.figure(figsize=(8,3))
            plt.subplot(1,2,1)
            plt.title('Left eye')
            plt.imshow(inp_left[0,...,0], cmap="gray")
            plt.subplot(1,2,2)
            plt.title('Right eye')
            plt.imshow(inp_right[0,...,0], cmap="gray")
            plt.show()
            
            """## Predict eye region landmarks

            ELG forwardpass. Output shape: (36, 60, 18)
            """

            inp_left.shape, inp_right.shape

            input_array = np.concatenate([inp_left, inp_right], axis=0)
            pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

            """## Visualize output heatmaps

            Eighteen heatmaps are predicted: 
            - Eight heatmaps for iris (green)
            - Eight heatmaps for eye-lid (red)
            - Two heatmaps for pupil (blue)
            """

            #@title
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.axis('off')
            plt.title('Left eye heatmaps')
            hm_r = np.max(pred_left[...,:8], axis=-1, keepdims=True)
            hm_g = np.max(pred_left[...,8:16], axis=-1, keepdims=True)
            hm_b = np.max(pred_left[...,16:], axis=-1, keepdims=True)
            plt.imshow(np.concatenate([hm_r, hm_g, hm_b], axis=-1))
            plt.subplot(1,2,2)
            plt.axis('off')
            plt.title('Right eye heatmaps')
            hm_r = np.max(pred_right[...,:8], axis=-1, keepdims=True)
            hm_g = np.max(pred_right[...,8:16], axis=-1, keepdims=True)
            hm_b = np.max(pred_right[...,16:], axis=-1, keepdims=True)
            plt.imshow(np.concatenate([hm_r, hm_g, hm_b], axis=-1))
            plt.show()
            
            """# Draw eye region landmarks"""

            def draw_pupil(im, inp_im, lms):
                draw = im.copy()
                draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
                pupil_center = np.zeros((2,))
                pnts_outerline = []
                pnts_innerline = []
                stroke = inp_im.shape[1] // 12 + 1
                
                for i, lm in enumerate(np.squeeze(lms)):
                    y, x = int(lm[0]*3), int(lm[1]*3)

                    if i < 8:
                        pnts_outerline.append([y, x])
                    elif i < 16:
                        pnts_innerline.append([y, x])
                        pupil_center += (y,x)
                
                pupil_center = (pupil_center/8).astype(np.int32)
                draw = cv2.cv2.circle(draw, (pupil_center[0], pupil_center[1]), 1, (255,255,0), -1)        
                draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(125,255,125), thickness=stroke//4)
                draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(125,125,255), thickness=stroke//4)
                return draw, pupil_center

            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            plt.title("Left eye")
            lms_left = model._calculate_landmarks(pred_left)
            result_left, pupil_center_left = draw_pupil(left_eye_im, inp_left, lms_left)
            plt.imshow(result_left)
            #print(pupil_center_left)

            plt.subplot(1,2,2)
            plt.title("Right eye")
            lms_right = model._calculate_landmarks(pred_right)
            result_right, pupil_center_right = draw_pupil(right_eye_im, inp_right, lms_right)
            plt.imshow(result_right)
            #print(pupil_center_right)
            plt.show()

            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            ratio_x1=dim_x2/180
            #print(ratio_x1)
            #print(pupil_center_left[0])
            pupil_center_left[0]=int((pupil_center_left[0])*(ratio_x1))
            ratio_y1=dim_y2/108
            pupil_center_left[1]=int(pupil_center_left[1]*ratio_y1)
            pupil_center_left_cp = cv2.cv2.circle(pupil_center_left_cp, (pupil_center_left[0], pupil_center_left[1]), 10, (255,255,0), -1)
            print(pupil_center_left)
            plt.imshow(pupil_center_left_cp)

            plt.subplot(1,2,2)
            ratio_x2=dim_x1/180
            ratio_y2=dim_y1/108
            pupil_center_right[0]=int(pupil_center_right[0]*ratio_x2)
            pupil_center_right[1]=int(pupil_center_right[1]*ratio_y2)
            pupil_center_right_cp = cv2.cv2.circle(pupil_center_right_cp, (pupil_center_right[0], pupil_center_right[1]), 10, (255,255,0), -1)
            print(pupil_center_right)
            plt.imshow(pupil_center_right_cp)
            
            dist_bet_centre = np.linalg.norm(pupil_center_left - pupil_center_right)
            print(dist_bet_centre)
            print("pixel")
            measured=int(self.lastName.text)
            max_x_res=int(self.xres.text)
            print(round(dist_bet_centre*(measured/max_x_res),2))
            print("in mm")

            if cnt_one==0:
                cv2.imwrite(img_name_lf, pupil_center_left_cp)
                cv2.imwrite(img_name_rt, pupil_center_right_cp)
            if cnt_one==1:
                cv2.imwrite(img_name_md, pupil_center_right_cp)

            pnt_center=[]
            
            #plt.show()

##            if cnt_one==0:
##                a=str(round(dist_bet_centre*(measured/max_x_res),2))
##            elif cnt_one==1:
##                b=str(round(dist_bet_centre*(measured/max_x_res),2))
##            elif cnt_one==2:
##                c=str(round(dist_bet_centre*(measured/max_x_res),2))
##                data['dist'].append({  
##                '1': str(a),
##                '2': str(b),
##                '3': str(c)
##                })

            cnt_one+=1
        

    def pressedrunleft(self, instance):
        name = self.name.text
        dist_both = self.email.text

        print("Name:", name)


        """## Instantiate GazeML ELG model

        #A stacked hourglass framework for iris and eye-lid detection.
        """

        model = KerasELG()
        model.net.load_weights("./elg_weights/elg_keras.h5")

        cnt_one_lf=0
        while cnt_one_lf<3:
            
            if cnt_one_lf==0:
                fnr = "./one/left/left1.jpg"
                right_eye_im = cv2.imread(fnr)[..., ::-1]
                dim_x1=right_eye_im.shape[1]
                dim_y1=right_eye_im.shape[0]
                
                fnl = "./one/left/left2.jpg"
                left_eye_im = cv2.imread(fnl)[..., ::-1]
                dim_x2=left_eye_im.shape[1]
                dim_y2=left_eye_im.shape[0]

                makeanim(right_eye_im , left_eye_im , './anim/one/left_rt_lf' )
            
            if cnt_one_lf==1:
                fnr = "./one/left/left2.jpg"
                right_eye_im = cv2.imread(fnr)[..., ::-1]
                dim_x1=right_eye_im.shape[1]
                dim_y1=right_eye_im.shape[0]
                
                fnl = "./one/left/leftmid.jpg"
                left_eye_im = cv2.imread(fnl)[..., ::-1]
                dim_x2=left_eye_im.shape[1]
                dim_y2=left_eye_im.shape[0]

                makeanim(right_eye_im , left_eye_im , './anim/one/left_lf_md')
            if cnt_one_lf==2:
                fnr = "./one/left/left1.jpg"
                right_eye_im = cv2.imread(fnr)[..., ::-1]
                dim_x1=right_eye_im.shape[1]
                dim_y1=right_eye_im.shape[0]
                
                fnl = "./one/left/leftmid.jpg"
                left_eye_im = cv2.imread(fnl)[..., ::-1]
                dim_x2=left_eye_im.shape[1]
                dim_y2=left_eye_im.shape[0]
                makeanim(right_eye_im , left_eye_im , './anim/one/left_rt_md' )
                
            pupil_center_right_cp=right_eye_im.copy()
            pupil_center_left_cp=left_eye_im.copy()



            plt.subplot(1,2,1)
            plt.title('Input image rt')
            plt.imshow(right_eye_im)
            plt.subplot(1,2,2)
            plt.title('Input image lf')
            plt.imshow(left_eye_im)
            plt.show()
            
            inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
            inp_left = cv2.equalizeHist(inp_left)
            inp_left = cv2.resize(inp_left, (180,108))[np.newaxis, ..., np.newaxis]

            inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
            inp_right = cv2.equalizeHist(inp_right)
            inp_right = cv2.resize(inp_right, (180,108))[np.newaxis, ..., np.newaxis]

            plt.figure(figsize=(8,3))
            plt.subplot(1,2,1)
            plt.title('Left eye')
            plt.imshow(inp_left[0,...,0], cmap="gray")
            plt.subplot(1,2,2)
            plt.title('Right eye')
            plt.imshow(inp_right[0,...,0], cmap="gray")
            plt.show()

            """## Predict eye region landmarks

            ELG forwardpass. Output shape: (36, 60, 18)
            """

            inp_left.shape, inp_right.shape

            input_array = np.concatenate([inp_left, inp_right], axis=0)
            pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

            """## Visualize output heatmaps

            Eighteen heatmaps are predicted: 
            - Eight heatmaps for iris (green)
            - Eight heatmaps for eye-lid (red)
            - Two heatmaps for pupil (blue)
            """

            #@title
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.axis('off')
            plt.title('Left eye heatmaps')
            hm_r = np.max(pred_left[...,:8], axis=-1, keepdims=True)
            hm_g = np.max(pred_left[...,8:16], axis=-1, keepdims=True)
            hm_b = np.max(pred_left[...,16:], axis=-1, keepdims=True)
            plt.imshow(np.concatenate([hm_r, hm_g, hm_b], axis=-1))
            plt.subplot(1,2,2)
            plt.axis('off')
            plt.title('Right eye heatmaps')
            hm_r = np.max(pred_right[...,:8], axis=-1, keepdims=True)
            hm_g = np.max(pred_right[...,8:16], axis=-1, keepdims=True)
            hm_b = np.max(pred_right[...,16:], axis=-1, keepdims=True)
            plt.imshow(np.concatenate([hm_r, hm_g, hm_b], axis=-1))
            plt.show()

            """# Draw eye region landmarks"""

            def draw_pupil(im, inp_im, lms):
                draw = im.copy()
                draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
                pupil_center = np.zeros((2,))
                pnts_outerline = []
                pnts_innerline = []
                stroke = inp_im.shape[1] // 12 + 1
                
                for i, lm in enumerate(np.squeeze(lms)):
                    y, x = int(lm[0]*3), int(lm[1]*3)

                    if i < 8:
                        pnts_outerline.append([y, x])
                    elif i < 16:
                        pnts_innerline.append([y, x])
                        pupil_center += (y,x)
                
                pupil_center = (pupil_center/8).astype(np.int32)
                draw = cv2.cv2.circle(draw, (pupil_center[0], pupil_center[1]), 1, (255,255,0), -1)        
                draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(125,255,125), thickness=stroke//4)
                draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(125,125,255), thickness=stroke//4)
                return draw, pupil_center

            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            plt.title("Left eye")
            lms_left = model._calculate_landmarks(pred_left)
            result_left, pupil_center_left = draw_pupil(left_eye_im, inp_left, lms_left)
            plt.imshow(result_left)
            print(pupil_center_left)

            plt.subplot(1,2,2)
            plt.title("Right eye")
            lms_right = model._calculate_landmarks(pred_right)
            result_right, pupil_center_right = draw_pupil(right_eye_im, inp_right, lms_right)
            plt.imshow(result_right)
            print(pupil_center_right)
            plt.show()

            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            ratio_x1=dim_x2/180
            print(ratio_x1)
            print(pupil_center_left[0])
            pupil_center_left[0]=int((pupil_center_left[0])*(ratio_x1))
            ratio_y1=dim_y2/108
            pupil_center_left[1]=int(pupil_center_left[1]*ratio_y1)
            pupil_center_left_cp = cv2.cv2.circle(pupil_center_left_cp, (pupil_center_left[0], pupil_center_left[1]), 10, (255,255,0), -1)
            print(pupil_center_left)
            plt.imshow(pupil_center_left_cp)

            plt.subplot(1,2,2)
            ratio_x2=dim_x1/180
            ratio_y2=dim_y1/108
            pupil_center_right[0]=int(pupil_center_right[0]*ratio_x2)
            pupil_center_right[1]=int(pupil_center_right[1]*ratio_y2)
            pupil_center_right_cp = cv2.cv2.circle(pupil_center_right_cp, (pupil_center_right[0], pupil_center_right[1]), 10, (255,255,0), -1)
            print(pupil_center_right)
            plt.imshow(pupil_center_right_cp)

            dist_bet_centre = np.linalg.norm(pupil_center_left - pupil_center_right)
            print(dist_bet_centre)
            print("pixel")
            measured=int(self.lastName.text)
            max_x_res=int(self.xres.text)
            print(round(dist_bet_centre*(measured/max_x_res),2))

            pnt_center=[]
            #plt.show()
            
##            if cnt_one_lf==0:
##                a=str(round(dist_bet_centre*(measured/max_x_res),2))
##            elif cnt_one_lf==1:
##                b=str(round(dist_bet_centre*(measured/max_x_res),2))
##            elif cnt_one_lf==2:
##                c=str(round(dist_bet_centre*(measured/max_x_res),2))
##                data['dist'].append({  
##                '1': str(a),
##                '2': str(b),
##                '3': str(b)
##                })

            cnt_one_lf+=1
        
    
    def pressedrtw(self, instance):
        ##specify which camera(button)
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Camera")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera", frame)
            if not ret:
                break

            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "./one/right/rightmid.jpg"
                cv2.imwrite(img_name, frame)
                print("written!".format(img_name))
                break
        cam.release()

        cv2.destroyWindow('Camera')
            
    def pressedrtq(self, instance):
        ##specify which camera(button)
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Camera")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera", frame)
            if not ret:
                break

            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "./one/right/right1.jpg"
                cv2.imwrite(img_name, frame)
                print("written!".format(img_name))
                break
        cam.release()

        cv2.destroyWindow('Camera')
        
    def pressedrte(self, instance):
        ##specify which camera(button)
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Camera")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera", frame)
            if not ret:
                break

            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "./one/right/right2.jpg"
                cv2.imwrite(img_name, frame)
                print("written!".format(img_name))
                break
        cam.release()

        cv2.destroyWindow('Camera')

    def pressedlfw(self, instance):
        ##specify which camera(button)
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Camera")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera", frame)
            if not ret:
                break

            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "./one/left/leftmid.jpg"
                cv2.imwrite(img_name, frame)
                print("written!".format(img_name))
                break
        cam.release()

        cv2.destroyWindow('Camera')

    def pressedlfq(self, instance):
            ##specify which camera(button)
            cam = cv2.VideoCapture(0)

            cv2.namedWindow("Camera")

            while True:
                ret, frame = cam.read()
                cv2.imshow("Camera", frame)
                if not ret:
                    break

                k = cv2.waitKey(1)

                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "./one/left/left1.jpg"
                    cv2.imwrite(img_name, frame)
                    print("written!")
                    break
                
            cam.release()

            cv2.destroyWindow('Camera')

    def pressedlfe(self, instance):
            ##specify which camera(button)
            cam = cv2.VideoCapture(0)

            cv2.namedWindow("Camera")

            while True:
                ret, frame = cam.read()
                cv2.imshow("Camera", frame)
                if not ret:
                    break

                k = cv2.waitKey(1)

                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "./one/left/left2.jpg"
                    cv2.imwrite(img_name, frame)
                    print("written!")
                    break
                
            cam.release()

            cv2.destroyWindow('Camera')

    def pressedbmd(self, instance):
        ##specify which camera(button)
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Camera")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera", frame)
            if not ret:
                break

            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "./both/bmd.jpg"
                cv2.imwrite(img_name, frame)
                print("written!".format(img_name))
                break
        cam.release()

        cv2.destroyWindow('Camera')
        
    def pressedbrt(self, instance):
        ##specify which camera(button)
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Camera")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera", frame)
            if not ret:
                break

            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "./both/brt.jpg"
                cv2.imwrite(img_name, frame)
                print("written!".format(img_name))
                break
        cam.release()

        cv2.destroyWindow('Camera')

    def pressedblf(self, instance):
        ##specify which camera(button)
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Camera")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera", frame)
            if not ret:
                break

            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "./both/blf.jpg"
                cv2.imwrite(img_name, frame)
                print("written!".format(img_name))
                break
        cam.release()

        cv2.destroyWindow('Camera')
    ##################################################################################
        """Progress Bar VAlue set"""
    def trigger(self, number):
        """ Start the list of tasks """
        # The tasklist can be any list of functions. The only requirement is
        # that each accepts 1 parameter - callback to call when done
        task_list = [TaskSupplier.do_task for num in range(0, number)]
        ProgressAnimator(self.pb, task_list)
    ##################################################################################   

class Eye_Squint_Detector(App):
    def build(self):
        return MyGrid()


if __name__ == "__main__":
    Eye_Squint_Detector().run()
