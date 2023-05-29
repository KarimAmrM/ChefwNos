from apiclient.discovery import build
youtube = build('youtube','v3',developerKey = "AIzaSyDGAQyJkoinKG3WbujsU0W5slbHSdWdJpA")
print(type(youtube))

request = youtube.search().list(q='وصفة ميلك شيك التوت',part='snippet',type='video', videoDuration='short')

res = request.execute()
video_id = res['items'][0]['id']['videoId']
print(video_id)

#download video
from pytube import YouTube

YouTube('https://www.youtube.com/watch?v='+video_id).streams.first().download()
yt = YouTube('https://www.youtube.com/watch?v='+video_id)
yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()