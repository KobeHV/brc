import numpy.ma as npm
import cv2
import numpy as np
#
# img = cv2.imread("captured/2.jpg")
# threshold = 120
# mask = (img[:, :, 0] > threshold) & (img[:, :, 1] > threshold) & (img[:, :, 2] > threshold)
# # mask
# img[mask] = 0
#
# cv2.imshow('mask', img)
# # print(img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # data = np.random.randint(0, 10,, 3, size = [1, 5, 5])
# mask = data < 5
# arr = npm.array(data, mask=mask)
# print(arr)

# [[[6 6 -- 8 --]
#  [-- -- -- 6 7]
#  [9 -- -- 6 9]
#  [-- -- 5 -- 8]
#  [6 9 -- 5 --]]]


image_array = cv2.imread("./restore/white.png")

rows, cols, channels = image_array.shape
for i in range(rows):
    for j in range(cols):
        print(image_array[i][j][0], image_array[i][j][1], image_array[i][j][2])

# import os
#
# # for root, dirs, files in os.walk("D:/BRC-Project/FGC/datasets/final_new"):
# #     for dir_name in dirs:
# #         dir_pth = os.path.join(root, dir_name)
# #         for dir_root, _, dir_files in os.walk(dir_pth):
# #             print(dir_name, "len:", len(dir_files))
# #             for i, name in enumerate(dir_files):
# #                 file_pth = os.path.join(dir_root, name)
# #                 new_name = dir_name + "_" + str(i) + ".jpg"
# #                 os.rename(file_pth, os.path.join(dir_root, new_name))
# #                 print(i, "-", file_pth)
# for root, dirs, files in os.walk("D:/BRC-Project/FGC/datasets/final_new"):
#     for name in files:
#         print(os.path.join(root, name))
#
# img = cv2.imread("captured/2.jpg")
# size = img.shape
# print(size)
start = time.time()
end = time.time()
print('Running time: %s Seconds' % (end - start))