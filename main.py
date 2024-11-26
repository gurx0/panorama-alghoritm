import cv2
from pathlib import Path
import numpy as np


class VideMosaic:
    def __init__(self, first_image, output_height_times=2, output_width_times=2, detector_type="sift"):
        """This class processes every frame and generates the panorama

        Args:
            first_image (image for the first frame): first image to initialize the output size
            output_height_times (int, optional): determines the output height based on input image height. Defaults to 2.
            output_width_times (int, optional): determines the output width based on input image width. Defaults to 4.
            detector_type (str, optional): the detector for feature detection. It can be "sift" or "orb". Defaults to "sift".
        """

        print(f"detector_type - {detector_type}")
        self.detector_type = detector_type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(1000)
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(1000)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.visualize = True

        self.process_first_frame(first_image)

        self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), int(
            output_width_times*first_image.shape[1]), first_image.shape[2]))

        # offset
        self.w_offset = int(self.output_img.shape[0]/2 - first_image.shape[0]/2)
        self.h_offset = int(self.output_img.shape[1]/2 - first_image.shape[1]/2)

        self.output_img[self.w_offset:self.w_offset+first_image.shape[0],
                        self.h_offset:self.h_offset+first_image.shape[1], :] = first_image

        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

    def process_first_frame(self, first_image):

        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def match(self, des_cur, des_prev):

        if self.detector_type == "sift":
            pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
            matches = []
            for m, n in pair_matches:
                if m.distance < 0.7*n.distance:
                    matches.append(m)

        elif self.detector_type == "orb":
            matches = self.bf.match(des_cur, des_prev)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # get the maximum of 20  best matches
        matches = matches[:min(len(matches), 20)]
        # Draw first 10 matches.
        if self.visualize:
            match_img = cv2.drawMatches(self.frame_cur, self.kp_cur, self.frame_prev, self.kp_prev, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matches', match_img)
        return matches

    def process_frame(self, frame_cur):
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        if self.kp_cur is None or len(self.kp_cur) < 10:
            print("Not enough keypoints detected, skipping frame.")
            return  # Пропускаем кадр, если ключевых точек мало

        self.matches = self.match(self.des_cur, self.des_prev)

        if len(self.matches) < 4:
            print("Not enough matches, skipping frame.")
            return  # Пропускаем кадр, если совпадений мало

        # Найти гомографию
        self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)

        # Проверяем адекватность гомографии
        if self.H is None or not self.is_valid_homography(self.H):
            print("Invalid homography detected, skipping frame.")
            return  # Пропускаем кадр, если гомография некорректна

        # Проверка размера кадра
        if not self.is_valid_scale(self.H):
            print("Frame scale change exceeds limit, skipping frame.")
            return  # Пропускаем кадр, если изменение размера превышает предел

        self.H = np.matmul(self.H_old, self.H)
        self.warp(self.frame_cur, self.H)

        # Подготовка к следующему кадру
        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur

    @ staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):

        # taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        homography, mask = cv2.findHomography(
            image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

        return homography

    def warp(self, frame_cur, H):

        warped_img = cv2.warpPerspective(
            frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR)

        transformed_corners = self.get_transformed_corners(frame_cur, H)
        min_x = transformed_corners[:, :, 0].min()
        max_x = transformed_corners[:, :, 0].max()
        min_y = transformed_corners[:, :, 1].min()
        max_y = transformed_corners[:, :, 1].max()

        if min_x < 0 or max_x >= self.output_img.shape[1] or min_y < 0 or max_y >= self.output_img.shape[0]:
            self.expand_canvas(min_x, max_x, min_y, max_y)
            warped_img = cv2.warpPerspective(
                frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR)

        warped_img = self.draw_border(warped_img, transformed_corners)
        self.output_img[warped_img > 0] = warped_img[warped_img > 0]

        output_temp = np.copy(self.output_img)
        output_temp = self.draw_border(output_temp, transformed_corners, color=(0, 0, 255))

        cv2.imshow('output', output_temp / 255.)

        cv2.imwrite('mosaic2.jpg', self.output_img.astype(np.uint8))

        return self.output_img

    def expand_canvas(self, min_x, max_x, min_y, max_y):

        new_width = int(max(max_x, self.output_img.shape[1]) - min(min_x, 0))
        new_height = int(max(max_y, self.output_img.shape[0]) - min(min_y, 0))

        x_offset = int(-min(min_x, 0))
        y_offset = int(-min(min_y, 0))

        new_canvas = np.zeros((new_height, new_width, 3), dtype=self.output_img.dtype)

        new_canvas[y_offset:y_offset + self.output_img.shape[0],
        x_offset:x_offset + self.output_img.shape[1]] = self.output_img

        translation_matrix = np.array([[1, 0, x_offset],
                                       [0, 1, y_offset],
                                       [0, 0, 1]], dtype=float)
        self.H = translation_matrix @ self.H

        self.output_img = new_canvas

    @ staticmethod
    def get_transformed_corners(frame_cur, H):

        corner_0 = np.array([0, 0])
        corner_1 = np.array([frame_cur.shape[1], 0])
        corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner_3 = np.array([0, frame_cur.shape[0]])

        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = np.array(transformed_corners, dtype=np.int32)

        # mask = np.zeros(shape=(output.shape[0], output.shape[1], 1))
        # cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
        # cv2.imshow('mask', mask)

        return transformed_corners

    def draw_border(self, image, corners, color=(0, 0, 0)):

        for i in range(corners.shape[1]-1, -1, -1):
            cv2.line(image, tuple(corners[0, i, :]), tuple(
                corners[0, i-1, :]), thickness=5, color=color)
        return image

    @staticmethod
    def is_valid_homography(H):
        if not np.isfinite(H).all():
            return False
        det = np.linalg.det(H[:2, :2])
        if det < 0.1 or det > 10:
            return False
        if np.abs(H[0, 2]) > 500 or np.abs(H[1, 2]) > 500:
            return False
        return True

    def is_valid_scale(self, H):
        # Векторы осей после трансформации
        scale_x = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        scale_y = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)

        # Допустимые пределы изменения масштаба (±10%)
        min_scale = 0.9
        max_scale = 1.1

        # Проверяем, укладываются ли масштабы в заданные пределы
        if not (min_scale <= scale_x <= max_scale):
            print(f"Scale X out of bounds: {scale_x}")
            return False
        if not (min_scale <= scale_y <= max_scale):
            print(f"Scale Y out of bounds: {scale_y}")
            return False

        return True


def main():
    # fragment = cv2.imread('img.jpg')
    video_path = 'Data/DJI_0183.MP4'
    cap = cv2.VideoCapture(video_path)
    is_first_frame = True
    cap.read()
    while cap.isOpened():

        ret, frame_cur = cap.read()
        frame_cur = cv2.resize(frame_cur, (640, 480))
        if not ret:
            if is_first_frame:
                continue
            break

        if is_first_frame:
            # frame_cur = cv2.resize(frame_cur, (1920, 1080))
            video_mosaic = VideMosaic(frame_cur, detector_type="sift")
            # video_mosaic.find_and_draw_fragment(fragment)

            is_first_frame = False
            continue

        try:
            video_mosaic.process_frame(frame_cur)
        except Exception as e:
            print(f"Error processing frame: {e}")


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()