"""
将图像进行切分,将图像切分成不同规格的小图片,并将其对应的label进行修改调整.

"""
import os
import cv2
import numpy as np
import math
from parse_tal_xml import ParseXml, paser_text

label_temp_dir = 'train_label_tmp'
out_path = 'train_img_tmp'
bbox_out_dir = 'train_bbbox_tmp'
if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(label_temp_dir):
    os.makedirs(label_temp_dir)
if not os.path.exists(bbox_out_dir):
    os.makedirs(bbox_out_dir)

def draw_bbox(img, class_list, bbox_list, save_name):
    print_color = (0, 0, 255)
    hand_color = (255, 0, 0)

    for i in range(len(class_list)):
        if class_list[i] == 0:
            color = print_color
        else:
            color = hand_color

        if len(bbox_list[i]) == 4:
            cv2.rectangle(img, (bbox_list[i][0], bbox_list[i][1]), (bbox_list[i][2], bbox_list[i][3]), color, 2)
        else:
            cv2.line(img, (bbox_list[i][0], bbox_list[i][1]),
                     (bbox_list[i][2], bbox_list[i][3]), color, 2)
            cv2.line(img, (bbox_list[i][2], bbox_list[i][3]),
                     (bbox_list[i][4], bbox_list[i][5]), color, 2)
            cv2.line(img, (bbox_list[i][4], bbox_list[i][5]),
                     (bbox_list[i][6], bbox_list[i][7]), color, 2)
            cv2.line(img, (bbox_list[i][6], bbox_list[i][7]),
                     (bbox_list[i][0], bbox_list[i][1]), color, 2)
    #cv2.imwrite(save_name, img)
    return img


def clip_img(img_path, label_path):

    def find_clip_dot(line):
        dots_index = np.where(line==0)
        length = len(dots_index)
        midle = int(len(line)/2)
        dots_index = dots_index[0]
        distance = abs(dots_index-midle)

        dot = dots_index[np.where(distance == min(distance))]
        return dot

    def clip_img_labels(img, bboxes, point, type):
        if type == 1:
            cut_img = img[0:point[0]+1, 0:point[1]+1,:]
            res_bboxes = bboxes
        elif type == 2:
            cut_img = img[0:point[0], point[1]:, :]
            res_bboxes = []
            for bbox in bboxes:
                if len(bbox) == 4:
                    bbox[0] -= point[1]
                    bbox[2] -= point[1]
                else:
                    bbox[0] -= point[1]
                    bbox[2] -= point[1]
                    bbox[4] -= point[1]
                    bbox[6] -= point[1]
                res_bboxes.append(bbox)
        elif type == 3:
            cut_img = img[point[0]:, 0:point[1]+1, :]
            res_bboxes = []
            for bbox in bboxes:
                if len(bbox) == 4:
                    bbox[1] -= point[0]
                    bbox[3] -= point[0]
                else:
                    bbox[1] -= point[0]
                    bbox[3] -= point[0]
                    bbox[5] -= point[0]
                    bbox[7] -= point[0]
                res_bboxes.append(bbox)
        elif type == 4:
            cut_img = img[point[0]:, point[1]:, :]
            res_bboxes = []
            for bbox in bboxes:
                if len(bbox) == 4:
                    bbox[1] -= point[0]
                    bbox[3] -= point[0]
                    bbox[0] -= point[1]
                    bbox[2] -= point[1]
                else:
                    bbox[1] -= point[0]
                    bbox[3] -= point[0]
                    bbox[5] -= point[0]
                    bbox[7] -= point[0]
                    bbox[0] -= point[1]
                    bbox[2] -= point[1]
                    bbox[4] -= point[1]
                    bbox[6] -= point[1]
                res_bboxes.append(bbox)

        return cut_img, res_bboxes
    print(img_path)
    img = cv2.imread(img_path)

    if 'xml' in label_path:
        is_xml = True
    elif 'txt' in label_path:
        is_xml = False
    else:
        assert 0,'{} error'.format(label_path)

    if is_xml:
        p = ParseXml(label_path)
        img_name, classes, bboxes = p.get_bbox_class()
    else:
        classes, bboxes = paser_text(label_path)

    # for b in bboxes:
    #     cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),(255,0,0))
    #
    # cv2.imshow("wd",img)
    # cv2.waitKey()

    h, w, _ = img.shape

    h_flag = np.zeros([h])
    w_flag = np.zeros([w])

    for bbox in bboxes:
        if len(bbox) == 4:
            # print(bbox[1],bbox[3],bbox[0],bbox[2])
            h_flag[bbox[1]:bbox[3]+1] = 1
            w_flag[bbox[0]:bbox[2]+1] = 1
        elif len(bbox) == 8:
            h_flag[min(bbox[1], bbox[3], bbox[5], bbox[7]):max(bbox[1], bbox[3], bbox[5], bbox[7])+1] = 1
            w_flag[min(bbox[0], bbox[2], bbox[4], bbox[6]):max(bbox[0], bbox[2], bbox[4], bbox[6])+1] = 1

    cut_h_index = find_clip_dot(h_flag)[0]
    cut_w_index = find_clip_dot(w_flag)[0]

    bboxes_top_l = []
    bboxes_top_r = []
    bboxes_bot_l = []
    bboxes_bot_r = []

    classes_top_l = []
    classes_top_r = []
    classes_bot_l = []
    classes_bot_r = []

    for i, bbox in enumerate(bboxes):
        # 先分为上下两部分
        if bbox[1] < cut_h_index:
            if bbox[0] < cut_w_index:
                bboxes_top_l.append(bbox)
                classes_top_l.append(classes[i])
            else:
                bboxes_top_r.append(bbox)
                classes_top_r.append(classes[i])
        else:
            if bbox[0] < cut_w_index:
                bboxes_bot_l.append(bbox)
                classes_bot_l.append(classes[i])
            else:
                bboxes_bot_r.append(bbox)
                classes_bot_r.append(classes[i])

    point = [cut_h_index,cut_w_index]
    cut_dataset_list = []

    num = 3
    if len(bboxes_top_l)>num:
        data_dict = {}
        cut_img, res_bboxes = clip_img_labels(img.copy(), bboxes_top_l, point, 1)
        data_dict["img"] = cut_img
        data_dict["class"] = classes_top_l
        data_dict["bbox"] = res_bboxes
        cut_dataset_list.append(data_dict)
        # for b in res_bboxes:
        #     cv2.rectangle(cut_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0))
        #
        # cv2.imshow("1", cut_img)
        # cv2.waitKey()

    if len(bboxes_top_r)>num:
        data_dict = {}
        cut_img, res_bboxes = clip_img_labels(img.copy(), bboxes_top_r, point, 2)
        # draw_bbox(cut_img, classes_top_r, res_bboxes, "_2.jpg")
        data_dict["img"] = cut_img
        data_dict["class"] = classes_top_r
        data_dict["bbox"] = res_bboxes
        cut_dataset_list.append(data_dict)
        # for b in res_bboxes:
        #     cv2.rectangle(cut_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0))
        #
        # cv2.imshow("2", cut_img)
        # cv2.waitKey()

    if len(bboxes_bot_l)>num:
        data_dict = {}
        cut_img, res_bboxes = clip_img_labels(img.copy(), bboxes_bot_l, point, 3)
        # draw_bbox(cut_img, classes_bot_l, res_bboxes, "_3.jpg")
        data_dict["img"] = cut_img
        data_dict["class"] = classes_bot_l
        data_dict["bbox"] = res_bboxes
        cut_dataset_list.append(data_dict)
        # for b in res_bboxes:
        #     cv2.rectangle(cut_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0))
        #
        # cv2.imshow("3", cut_img)
        # cv2.waitKey()

    if len(bboxes_bot_r)>num:
        data_dict = {}
        cut_img, res_bboxes = clip_img_labels(img.copy(), bboxes_bot_r, point, 4)
        # draw_bbox(cut_img, classes_bot_r, res_bboxes, "_4.jpg")
        data_dict["img"] = cut_img
        data_dict["class"] = classes_bot_r
        data_dict["bbox"] = res_bboxes
        cut_dataset_list.append(data_dict)
        # for b in res_bboxes:
        #     cv2.rectangle(cut_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0))
        #
        # cv2.imshow("4", cut_img)
        # cv2.waitKey()

    return cut_dataset_list


def split_label(img, class_list, bbox_list, img_name,
                proposal_width=8.0,
                class_name=['dontcare', 'handwritten', 'print']):

    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)

    # 图像进行resize
    re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    re_size = re_im.shape
    cv2.imwrite(os.path.join(out_path, img_name) + '.jpg', re_im)
    # _, class_list, bbox_list

    assert len(class_list) == len(bbox_list), 'bbox和label不对应'

    res_bboxes = []
    res_classes = []

    for bbox_index in range(len(bbox_list)):

        if class_list[bbox_index] == 2:
            continue

        if len(bbox_list[bbox_index]) == 8:
            xmin = int(np.floor(float(
                min(bbox_list[bbox_index][0], bbox_list[bbox_index][2], bbox_list[bbox_index][4],
                    bbox_list[bbox_index][6])) / img_size[0] * re_size[0]))
            ymin = int(np.floor(float(
                min(bbox_list[bbox_index][1], bbox_list[bbox_index][3], bbox_list[bbox_index][5],
                    bbox_list[bbox_index][7])) / img_size[1] * re_size[1]))
            xmax = int(np.ceil(float(
                max(bbox_list[bbox_index][0], bbox_list[bbox_index][2], bbox_list[bbox_index][4],
                    bbox_list[bbox_index][6])) / img_size[0] * re_size[0]))
            ymax = int(np.ceil(float(
                max(bbox_list[bbox_index][1], bbox_list[bbox_index][3], bbox_list[bbox_index][5],
                    bbox_list[bbox_index][7])) / img_size[1] * re_size[1]))
        elif len(bbox_list[bbox_index]) == 4:
            xmin = int(np.floor(float(bbox_list[bbox_index][0]) / img_size[0] * re_size[0]))
            ymin = int(np.floor(float(bbox_list[bbox_index][1]) / img_size[1] * re_size[1]))
            xmax = int(np.ceil(float(bbox_list[bbox_index][2]) / img_size[0] * re_size[0]))
            ymax = int(np.ceil(float(bbox_list[bbox_index][3]) / img_size[1] * re_size[1]))
        else:
            # print(xml_file)
            assert 0, "{}bbox error".format(img_name)

        if xmin < 0:
            xmin = 0
        if xmax > re_size[1] - 1:
            xmax = re_size[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > re_size[0] - 1:
            ymax = re_size[0] - 1

        res_bboxes.append([xmin,ymin,xmax,ymax])
        res_classes.append(class_list[bbox_index])

        width = xmax - xmin + 1
        height = ymax - ymin + 1

        # TODO proposal 宽度
        step = proposal_width
        x_left = []
        x_right = []
        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / proposal_width) * proposal_width)
        if x_left_start == xmin:
            x_left_start = xmin + proposal_width
        for i in np.arange(x_left_start, xmax, proposal_width):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + proposal_width - 1)
        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)

        if not os.path.exists(label_temp_dir):
            os.makedirs(label_temp_dir)

        if class_list[bbox_index] == 0:  # 手写框
            current_class = class_name[class_list[bbox_index] + 1]
            color = (255, 0, 0)
        elif class_list[bbox_index] == 1:  # 打印框
            current_class = class_name[class_list[bbox_index] + 1]
            color = (0, 255, 0)
        else:
            assert 0, '不该出现其他类型的class:{}'.format(class_list[bbox_index])

        with open(os.path.join(label_temp_dir, img_name) + '.txt', 'a+') as f:
            for i in range(len(x_left)):
                f.writelines(current_class)
                f.writelines("\t")
                f.writelines(str(x_left[i]))
                f.writelines("\t")
                f.writelines(str(ymin))
                f.writelines("\t")
                f.writelines(str(x_right[i]))
                f.writelines("\t")
                f.writelines(str(ymax))
                f.writelines("\n")
    #             cv2.rectangle(re_im, (int(x_left[i]),int(ymin)), (int(x_right[i]),int(ymax)), color,1)
    # cv2.imshow('22', re_im)
    # cv2.waitKey()
    with open(os.path.join(bbox_out_dir, img_name + ".txt"), "w") as f:
        for i, bbox in enumerate(res_bboxes):
            f.writelines(str(res_classes[i]))
            f.writelines("\t")
            f.writelines(str(bbox[0]))
            f.writelines("\t")
            f.writelines(str(bbox[1]))
            f.writelines("\t")
            f.writelines(str(bbox[2]))
            f.writelines("\t")
            f.writelines(str(bbox[3]))
            f.writelines("\n")



if __name__ == "__main__":

    # img = cv2.imread("./data/img/ocr_3rd_7.jpg")
    # xml_path = "./data/xml/ocr_3rd_7.xml"
    # clip_img(img, xml_path)


    # img_dir = "/home/tony/ocr/ocr_dataset/tal_ocr_data_v2/img"
    # xml_dir = "/home/tony/ocr/ocr_dataset/tal_ocr_data_v2/xml"
    # txt_dir = "/home/tony/ocr/Data_Augmentation/1-4/train_bbbox_tmp"

    img_dir = "/home/tony/ocr/Data_Augmentation/1-4/train_img_tmp"
    xml_dir = "/home/tony/ocr/ocr_dataset/tal_ocr_data_v2/xml"
    txt_dir = "/home/tony/ocr/Data_Augmentation/1-4/train_bbbox_tmp"

    img_list = os.listdir(img_dir)

    for img_name in img_list:
        base_name = img_name.split('.')[0]
        type = img_name.split('.')[-1]
        if type in ["jpg","png","JPG","jpeg"]:
            cut_dataset_list = clip_img(os.path.join(img_dir, img_name), os.path.join(txt_dir, base_name+".txt"))
            for i, img_dict in enumerate(cut_dataset_list):
                cut_img_name = base_name + "_" + str(i)

                with open(os.path.join(bbox_out_dir,cut_img_name + ".txt"), "w") as f:
                    for i, bbox in enumerate(img_dict["bbox"]):
                        f.writelines(str(img_dict["class"][i]))
                        f.writelines("\t")
                        f.writelines(str(bbox[0]))
                        f.writelines("\t")
                        f.writelines(str(bbox[1]))
                        f.writelines("\t")
                        f.writelines(str(bbox[2]))
                        f.writelines("\t")
                        f.writelines(str(bbox[3]))
                        f.writelines("\n")

                split_label(img_dict["img"], img_dict["class"], img_dict["bbox"], cut_img_name)



