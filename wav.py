import os
import subprocess
from pydub import AudioSegment
import argparse
import sys
'''
截取一段视频
ffmpeg -ss 00:00:04.473 -t 00:00:01 -i demo.mp4  -c:v libx264 -c:a aac -strict experimental -b:a 98k audio.mp4
无损截音频
ffmpeg -i ./out/teddy.mp4  -vn -acodec copy  ./audio/teddy.m4a
无损截视频
ffmpeg  -i ./plutopr.mp4 -vcodec copy -acodec copy -ss 00:00:10 -to 00:00:15 ./cutout1.mp4 -y

视频解帧
'ffmpeg -i {0} -q:v 2 -f image2 {1}/%03d.png'.format(video_path,outfile)
'''

'''
视频转mp4格式
'''
import os

video_base = "r-3-4"
save_path = "result"
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i,name in enumerate(os.listdir(video_base)):
    path = os.path.join(video_base,name)
    cmd = "ffmpeg -i %s -c:v libx264  -preset veryslow -crf 16  %s/%i.mp4"%(path,save_path,i)
    os.system(cmd) 
    # print(cmd)

'''
图片转视频

ffmpeg -f image2 -i /home/ttwang/images/image%d.jpg  -vcodec libx264 -r 10  tt.mp4
'''


def main(args):
    video=args.video
    wav=args.wav
    part=args.part
    temp_mp3='./temp/temp.mp3'
    temp_video='./temp/output.mp4'
    temp_song='./temp/new_song.mp3'
    temp_norm='./temp/new_norm.mp3'
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
    if os.path.exists(temp_video):
        os.remove(temp_video)
    if os.path.exists(temp_song):
        os.remove(temp_song)
    #分离音频
    get_wav='ffmpeg -i %s -f mp3 -vn %s'%(video,temp_mp3)
    subprocess.call(get_wav,shell = True)

    #去掉视频音频
    move_wav='ffmpeg -i %s -an %s'%(video,temp_video)
    subprocess.call(move_wav,shell = True)

    #分割音频
    song=AudioSegment.from_mp3(temp_mp3)
    song1=song[:part[0]]
    song3=song[part[1] :]
    if 'mp3' in wav:
        song2=AudioSegment.from_mp3(wav)
    elif 'wav' in wav:
        song2=AudioSegment.from_mp3(wav)
    new_song=song1+song2+song3
    new_song.export(temp_song,format='mp3')

    #归一化视频
    norm="ffmpeg-normalize -f -nt rms -ar 16000 {0} -o {1}".format(temp_song,temp_norm)
    subprocess.call(norm,shell = True)
    #合成音频视频
    compose='ffmpeg -i %s -i %s %s'%(temp_song,temp_video,'./output/'+video.split('/')[-1])
    subprocess.call(compose,shell = True)
    
def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--video','-v', type=str,
                        help='The input video path')
    
    parser.add_argument('--wav','-w', type=str,
                        help='The input wav path')
    parser.add_argument('--part', '-p',nargs='+', type=int,
                        help='The insert part')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



