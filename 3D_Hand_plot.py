import cv2
import mediapipe as mp
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.animation


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
brightness = 10

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.legend()

def animate(i):
    success, img = cap.read()
    cap.set(10, brightness)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x_arr = []
            y_arr = []
            z_arr = []
            for id, lm in enumerate(handLms.landmark):
                x_arr.append(lm.x)
                y_arr.append(lm.y)
                z_arr.append(lm.z)

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print("id:",id," | cx,cy:", cx, cy)

            color = 'green'
            marker = 'o'
            linestyle = 'dashed'
            linestyle_solid = 'solid'
            linewidth = 2
            markersize = 12

            plt.cla()
            #Thumb
            ax.plot3D(x_arr[0:5], y_arr[0:5], z_arr[0:5],
                      color = 'sandybrown',
                      linewidth = 3)

            #Index
            ax.plot3D(x_arr[5:9], y_arr[5:9], z_arr[5:9],
                      color = 'aqua',
                      linewidth = 3)

            #Middle
            ax.plot3D(x_arr[9:13], y_arr[9:13], z_arr[9:13],
                      color = 'gold',
                      linewidth = 3
                      )
            #Ring
            ax.plot3D(x_arr[13:17], y_arr[13:17], z_arr[13:17],
                      color = 'lightcoral',
                      linewidth = 3
                      )
            #Pinky
            ax.plot3D(x_arr[17:21], y_arr[17:21], z_arr[17:21] ,
                      color = 'lightskyblue',
                      linewidth = 3
                      )

            ax.plot3D([x_arr[0],x_arr[17]], [y_arr[0],y_arr[17]], [z_arr[0],z_arr[17]] ,
                      color = 'darkseagreen',
                      linewidth = 3
                      )
            ax.plot3D([x_arr[0], x_arr[5]], [y_arr[0], y_arr[5]], [z_arr[0], z_arr[5]] ,
                      color = 'darkseagreen',
                      linewidth = 3
                      )
            ax.plot3D([x_arr[2], x_arr[5]], [y_arr[2], y_arr[5]], [z_arr[2], z_arr[5]],
                      color = 'darkseagreen',
                      linestyle = 'dashed',
                      linewidth = 3
                      )
            ax.plot3D(x_arr[5:18:4], y_arr[5:18:4], z_arr[5:18:4],
                      color = 'darkseagreen',
                      linestyle = 'dashed',
                      linewidth = 3
                      )

            ax.scatter3D(x_arr, y_arr, z_arr, 'ro')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            print("_________________")
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            cv2.imshow('Webcam', img)

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=2, interval=100, repeat=True)

plt.show()