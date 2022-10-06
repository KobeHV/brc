import _thread

from system_utils import *
import warnings
import scipy.io as sio
import numpy as np
import time

NEW_HEIGHT = 600
IMG_PTH = "./captured/huai_1.jpg"

def system_run():
    camera = cv2.VideoCapture(0)
    print(f"Camera 1 is open？ {camera.isOpened()}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow('Camera 1 Video Window', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    image_count = 1
    while True:
        ret_value, frame = camera.read()

        if not ret_value:
            print("Camera 1 initialization failed!")
            break

        cv2.imshow('Camera 1 Video Window', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Exit the system successfully!")
            break
        elif key == ord('c'):
            print(frame.shape)
            cv2.imwrite(DIR_PATH + f"captured/{image_count}.png", frame[200:800, 480:1900])
            print(f"Take a screenshot and save as {image_count}.png!")

            label_all_objects(original_image_path=f"captured/{image_count}.png")

            image_count += 1

    camera.release()
    cv2.destroyAllWindows()


def label_all_objects(original_image_path):
    original_image_name = os.path.splitext(os.path.basename(original_image_path))[0]
    _thread.start_new_thread(detect_and_label_objects,
                             (original_image_path, DIR_PATH + "removed_background/", original_image_name,
                              DIR_PATH + "cropped/",
                              original_image_name, DIR_PATH + "detected/", original_image_name,
                              DIR_PATH + "detected_and_labeled/",
                              original_image_name,))


# for testing
def label_all_objects_without_camera(original_image_path):
    original_image_name = os.path.splitext(os.path.basename(original_image_path))[0]
    list_res = detect_and_label_objects(original_image_path,
                                        DIR_PATH + "removed_background/", original_image_name,
                                        DIR_PATH + "cropped/", original_image_name,
                                        DIR_PATH + "detected/", original_image_name,
                                        DIR_PATH + "detected_and_labeled/", original_image_name)
    return list_res


def de_size(im_pth):
    im = cv2.imread(im_pth)
    print("img_original_size:", im.shape)
    # 重置图片大小，保持高、宽比例
    newHeight = NEW_HEIGHT
    newWidth = int(newHeight * (im.shape[1] / im.shape[0]))
    im = cv2.resize(im, (newWidth, newHeight))
    new_pth = DIR_PATH + "captured/" + os.path.splitext(os.path.basename(im_pth))[0] + "_resize.png"
    print(os.getcwd())
    cv2.imwrite(new_pth, im)
    return new_pth


def retrieval(imgs):
    img_query_list = []
    for img in imgs:
        image = inference(img)
        image = image.to(device)
        feature, _ = model(image)
        feature = feature.cpu().detach().numpy()
        del image
        img_query_list.append(feature)
        del feature
    # print(len(img_query_list))

    load_fn = DIR_PATH + 'restore/features_list.mat'
    load_data = sio.loadmat(load_fn)
    database_list = load_data["features_list"]

    img_query = np.array(img_query_list)
    img_query = np.squeeze(img_query, axis=1)
    database = np.squeeze(database_list, axis=1)
    img_query = np.ascontiguousarray(img_query, dtype=np.float32)
    database = np.ascontiguousarray(database, dtype=np.float32)
    # print(img_query.flags['C_CONTIGUOUS'], database.flags['C_CONTIGUOUS'])

    import faiss

    dim, measure = 2048, faiss.METRIC_L2
    # param = 'IVF100, PQ16'
    param = 'HNSW64'
    index = faiss.index_factory(dim, param, measure)
    # print(index.is_trained)  # 此时输出为False，因为倒排索引需要训练k-means，
    index.train(database)  # 因此需要先训练index，再add向量 index.add(xb)
    index.add(database)
    # print(index.ntotal)

    k = 1  # topK的K值
    D, I = index.search(img_query, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    # print(I)
    # print(D)
    return I


def detect(original_image_path):
    img_path = original_image_path
    new_pth = de_size(img_path)

    start = time.time()
    list_res = label_all_objects_without_camera(new_pth)
    end = time.time()
    print('label_all_objects_without_camera time: %s Seconds' % (end - start))
    # print(type(list_res), len(list_res))
    print(list_res)
    print(type(list_res[0][0]))

    start = time.time()
    # retrieval
    imgs = []
    for img in list_res:
        imgs.append(img[0])
    index = retrieval(imgs)
    print('retrieval time: %s Seconds' % (time.time() - start))

    # python加载.mat文件
    load_fn = DIR_PATH + 'restore/img_list.mat'
    load_data = sio.loadmat(load_fn)
    img_pth_list = load_data["img_list"]

    for i in range(len(list_res)):
        index_pth = img_pth_list[index[i][0]].strip()
        # print(index_pth)
        # img_pth = os.path.join(DIR_PATH + "../Dataset/retrieval_db/", index_pth)
        img_pth = os.path.join(DIR_PATH + "retrieval_db/", index_pth)
        # img = Image.open(img_pth)
        # list_res[i].append(img)
        list_res[i].append(img_pth)

    return list_res


if __name__ == '__main__':
    start = time.time()
    warnings.filterwarnings('ignore')
    # system_run()
    # img_pth = r"E:\Workspace\TCM-Quality\backend_dev\example_data\x.jpg"
    img_pth = IMG_PTH
    list_res = detect(img_pth)

    print(list_res)
    print(type(list_res[0][0]))

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
