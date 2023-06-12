import datetime
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
import cv2
import pickle as pkl
import os

current_frame_index = 0

def update_duration(event):
    """ updates the duration after finding the duration """
    duration = vid_player.video_info()["duration"]
    end_time["text"] = str(datetime.timedelta(seconds=duration))
    progress_slider["to"] = duration


def update_scale(event):
    """ updates the scale value """
    progress_value.set(vid_player.current_duration())


def load_video():
    """ loads the video """
    file_path = filedialog.askopenfilename()

    if file_path:
        vid_player.load(file_path)

        progress_slider.config(to=0, from_=0)
        play_pause_btn["text"] = "Play"
        progress_value.set(0)


def seek(value):
    """ used to seek a specific timeframe """
    vid_player.seek(int(value))


def skip(index: int, scene_frames: list, direction: str):
    """ skip seconds """    

        
    if direction == 'plus':
        #get closest frame to current duration that is greater than current duration
        index = min(range(len(scene_frames)), key=lambda i: abs(scene_frames[i]-vid_player.current_duration()))
        #if index is last frame in list then do nothing
        if index == len(scene_frames)-1:
            return
        else:
            index += 1
    elif direction == 'minus':
        #get closest frame to current duration that is less than current duration
        index = min(range(len(scene_frames)), key=lambda i: abs(scene_frames[i]-vid_player.current_duration()))
        #if index is first frame in list then do nothing
        if index == 0:
            return
        else:
            index -= 1
            
    vid_player.seek(int(scene_frames[index]))
    progress_value.set(int(scene_frames[index]))


    
def play_pause():
    """ pauses and plays """
    if vid_player.is_paused():
        vid_player.play()
        play_pause_btn["text"] = "Pause"

    else:
        vid_player.pause()
        play_pause_btn["text"] = "Play"


def video_ended(event):
    """ handle video ended """
    progress_slider.set(progress_slider["to"])
    play_pause_btn["text"] = "Play"
    progress_slider.set(0)


scene_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"scenes/scenes.pkl")
scenes = pkl.load(open(scene_path,"rb"))

#for each scene get the first frame and add it to the list
scene_frames = []
for scene in scenes:
    scene_frames.append(scene[0])


open_cv_video = cv2.VideoCapture("C:/Users/Fastora/Documents/GitHub/ChefwNos/SceneChange/recipe2.mp4")
count = 0
frame_time = {}
#count number of frames in video
while open_cv_video.isOpened():
    ret, frame = open_cv_video.read()
    if ret:
        count += 1
        #add frame number and time to dictionary
        frame_time[count] = open_cv_video.get(cv2.CAP_PROP_POS_MSEC)
    else:
        break


#convert frame number to time
for i in range(0,len(scene_frames)):
    scene_frames[i] = frame_time[scene_frames[i]]/1000
    
root = tk.Tk()
root.title("Tkinter media")

load_btn = tk.Button(root, text="Load", command=load_video)
load_btn.pack()

vid_player = TkinterVideo(scaled=True, master=root)
vid_player.pack(expand=True, fill="both")

play_pause_btn = tk.Button(root, text="Play", command=play_pause)
play_pause_btn.pack()

skip_plus_5sec = tk.Button(root, text="Skip to previous step", command=lambda: skip(current_frame_index, scene_frames, 'minus'))
skip_plus_5sec.pack(side="left")

start_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
start_time.pack(side="left")

progress_value = tk.IntVar(root)

progress_slider = tk.Scale(root, variable=progress_value, from_=0, to=0, orient="horizontal", command=seek)
# progress_slider.bind("<ButtonRelease-1>", seek)
progress_slider.pack(side="left", fill="x", expand=True)

end_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
end_time.pack(side="left")

vid_player.bind("<<Duration>>", update_duration)
vid_player.bind("<<SecondChanged>>", update_scale)
vid_player.bind("<<Ended>>", video_ended )

skip_plus_5sec = tk.Button(root, text="Skip to next step", command=lambda: skip(current_frame_index, scene_frames, 'plus'))
skip_plus_5sec.pack(side="left")

root.mainloop()