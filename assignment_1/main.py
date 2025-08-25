import cv2
import os

def print_image_information(image):

    height, width, channels = image.shape
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Size: {image.size}")
    print(f"Data type: {image.dtype}")

def save_camera_information(output_path="solutions/camera_outputs.txt"):

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.release()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"fps: {int(fps)}\n")
        f.write(f"width: {int(width)}\n")
        f.write(f"height: {int(height)}\n")

    print(f"Camera information saved to {output_path}")

def main():

    image = cv2.imread("lena.png")
    print_image_information(image)
    save_camera_information("solutions/camera_outputs.txt")

if __name__ == "__main__":
    main()