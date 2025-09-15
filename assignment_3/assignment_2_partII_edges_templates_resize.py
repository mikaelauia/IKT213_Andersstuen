import cv2
import numpy as np
import os


def create_output_folder():
    if not os.path.exists('output'):
        os.makedirs('output')


def sobel_edge_detection(image, image_name):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)

    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)

    sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

    sobel_combined = np.uint8(sobel_combined)

    output_path = os.path.join('output', f'sobel_edges_{image_name}')
    cv2.imwrite(output_path, sobel_combined)

    return sobel_combined


def canny_edge_detection(image, threshold_1=50, threshold_2=50, image_name=""):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blurred, threshold_1, threshold_2)

    output_path = os.path.join('output', f'canny_edges_{image_name}')
    cv2.imwrite(output_path, edges)

    return edges


def template_match(image, template, image_name="", template_name=""):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image.copy()

    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template.copy()

    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    threshold = 0.9

    loc = np.where(res >= threshold)

    if len(image.shape) == 3:
        result = image.copy()
    else:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    output_path = os.path.join('output', f'template_match_{image_name}_{template_name}')
    cv2.imwrite(output_path, result)

    return result


def resize(image, scale_factor, up_or_down, image_name=""):
    if up_or_down == 'down':
        resized = image.copy()
        for i in range(scale_factor):
            resized = cv2.pyrDown(resized)
    elif up_or_down == 'up':
        resized = image.copy()
        for i in range(scale_factor):
            resized = cv2.pyrUp(resized)
    else:
        raise ValueError("up_or_down must be either 'up' or 'down'")

    filename = f'resized_{up_or_down}_{scale_factor}_{image_name}'
    output_path = os.path.join('output', filename)
    cv2.imwrite(output_path, resized)

    return resized


def main():
    create_output_folder()

    lambo = cv2.imread('lambo.png')

    if lambo is None:
        return

    sobel_edges = sobel_edge_detection(lambo, 'lambo.jpg')
    canny_edges = canny_edge_detection(lambo, 50, 50, 'lambo.jpg')

    shapes = cv2.imread('shapes.png')
    template = cv2.imread('shapes_template.jpg')

    if shapes is not None and template is not None:
        matched = template_match(shapes, template, 'shapes.jpg', 'template.jpg')

    resized_down = resize(lambo, 2, 'down', 'lambo.jpg')
    resized_up = resize(lambo, 2, 'up', 'lambo.jpg')


if __name__ == "__main__":
    main()