import sys
import os
import cv2

input_dir = './input_img'
output_dir = './other_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#使用dlib自带的frontal_face_detector作为我们的特征提取器
#detector = dlib.get_frontal_face_detector()
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            # 从文件读取图片
            img = cv2.imread(img_path)
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测 dets为返回的结果
            dets =  haar.detectMultiScale(gray_img, 1.3, 5)

            #使用enumerate 函数遍历序列中的元素以及它们的下标
            #下标i即为人脸序号
            #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
            #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
            for f_x, f_y, f_w, f_h in dets:
                face = img[f_y:f_y + f_h, f_x:f_x + f_w]
                # 调整图片的尺寸
                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)
                # 保存图片
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)