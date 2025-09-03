import cv2
import numpy as np

image = cv2.imread('lena.png')


def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,
        cv2.BORDER_REFLECT
    )
    cv2.imwrite('lena_padded.png', padded_image)
    return padded_image


def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite('lena_cropped.png', cropped_image)
    return cropped_image


def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite('lena_resized.png', resized_image)
    return resized_image


def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]
    cv2.imwrite('lena_copied.png', emptyPictureArray)
    return emptyPictureArray


def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('lena_grayscale.png', gray_image)
    return gray_image


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite('lena_hsv.png', hsv_image)
    return hsv_image


def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    temp_image = image.astype(np.int16)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                new_value = temp_image[y, x, c] + hue
                new_value = max(0, min(255, new_value))
                emptyPictureArray[y, x, c] = new_value

    cv2.imwrite('lena_hue_shifted.png', emptyPictureArray)
    return emptyPictureArray

def hue_shifted_v2(image, emptyPictureArray, hue):
    temp_image = image.astype(np.int16)
    temp_image += hue
    temp_image = np.clip(temp_image, 0, 255)
    result = temp_image.astype(np.uint8)
    np.copyto(emptyPictureArray, result)

    cv2.imwrite('lena_hue_shifted.png', emptyPictureArray)
    return emptyPictureArray


def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite('lena_smoothed.png', smoothed_image)
    return smoothed_image


def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        print("Only 90 and 180 degree rotations are supported")
        return image

    cv2.imwrite('lena_rotated.png', rotated_image)
    return rotated_image


if __name__ == "__main__":

    padded = padding(image, 100)
    height, width = image.shape[:2]
    cropped = crop(image, 80, width - 130, 80, height - 130)
    resized = resize(image, 200, 200)
    empty_array = np.zeros_like(image)
    copied = copy(image, empty_array)
    gray = grayscale(image)
    hsv_img = hsv(image)
    empty_array2 = np.zeros_like(image)
    hue_shifted_img = hue_shifted(image, empty_array2, 50)
    smoothed = smoothing(image)
    rotated_90 = rotation(image, 90)
    rotated_180 = rotation(image, 180)

    print("All image processing operations completed successfully!")