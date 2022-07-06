import imageio
import cv2
import os

def create_gif(gif_name, pic_path, duration=0.3):
    frames = []
    images = os.listdir(pic_path)
    images.sort(key=lambda x:int(x.split('.')[0]))
    image_list = [os.path.join(pic_path, img) for img in images]
    for image_name in image_list:
        im = imageio.imread(image_name)
        frames.append(im)
    if frames:
        imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    else:
        print("gif gen failed")



def main():
    gif_name = 'SeamCarving/result/270result.gif'
    pic_path = 'SeamCarving/gif_frames/'   # 指定文件路径
    duration = 0.5
    create_gif(gif_name, pic_path, duration)


if __name__ == "__main__":
    main()
