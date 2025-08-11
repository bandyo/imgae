# 这段代码是为了处理排列比较整齐的种子图像,首先读取数据,然后选择一个波段用来做mask的提取，分割种子和背景。其次根据mask中各个种子的位置排序，最后将每个种子乘以对应区域的光谱数据，得到各个波长下的光谱反射率数值

# 导入高光谱数据提取、处理所需的库
import cv2
import numpy as np
from glob import glob
from pathlib import Path
import spectral.io.envi as envi
import csv
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import math
from datetime import datetime

# 1读取数据
# 定义一个名为GetWavelengths的函数，它接受一个参数headerPath
def GetWavelengths(headerPath):
    # 使用envi模块的read_envi_header函数读取指定路径headerPath的头文件
    # 并将读取到的头文件内容存储在变量header中
    header = envi.read_envi_header(headerPath)
    # 从header字典中获取键为'wavelength'的值（即波长数据）
    # 并将这个值赋给变量wavelengths
    wavelengths = header['wavelength']
    # 返回wavelengths变量，即波长数据
    return wavelengths
# 定义一个名为GetRawData的函数，它接受两个参数：
# headerPath - ENVI格式图像的头文件路径
# dataPath - ENVI格式图像的数据文件路径
def GetRawData(headerPath,dataPath):
    # 使用envi模块的open函数打开ENVI格式的图像文件
    # 该函数需要头文件路径和数据文件路径作为参数
    # 打开后的图像对象存储在变量image_data中
    image_data=envi.open(headerPath,dataPath)
    # 调用image_data对象的load()方法
    # 这个方法会将图像数据从磁盘加载到内存中
    # 加载后的数据仍然存储在image_data变量中
    image_data=image_data.load()
    # 返回加载到内存中的图像数据    
    return image_data
 
# 定义一个名为process_filename的函数，它接受三个参数：
# repeat - 包含重复信息的字符串，例如"gz_001-008-1.1_2025-06-15_01-57-11"中的"1.1"
# start - 起始数字，字符串形式
# end - 结束数字，字符串形式
# 这个函数处理特定格式的文件名，例如"gz_001-008-1.1_2025-06-15_01-57-11"
def process_filename(repeat, start ,end):
    # 从repeat字符串中提取位置排列方式
    # 如果repeat包含'.'，则取'.'前的部分并转换为整数
    # 如果不包含'.'，则默认位置排列方式为1
    position_type = int(repeat.split('.')[0] if '.' in repeat else 1 )
    # 生成基础数字序列
    # 将start和end字符串转换为整数
    start = int(start)
    end = int(end)
    # 生成从start到end(包含end)的数字列表
    numbers = list(range(start, end+1))
    # 根据位置排列方式调整顺序
    if position_type == 1:
        # 如果位置排列方式为1，保持数字顺序不变
        arranged = numbers
    elif position_type == 2:
        # 如果位置排列方式为2，将数字两两交换顺序
        arranged = []
        # 每次处理两个数字    
        for i in range(0, len(numbers), 2):
            # 如果还有下一个数字(i+1 < len(numbers))，则交换这两个数字的顺序
            if i+1 < len(numbers):
                arranged.extend([numbers[i+1], numbers[i]])
            else:
                # 如果是最后一个数字且总数为奇数，直接添加这个数字
                arranged.append(numbers[i])
    # 生成新的文件名列表
    # 将调整后的数字与原始repeat字符串组合，格式为"数字-repeat"
    new_filenames = [f"{i}-{repeat}" for i in arranged]
    # 返回新的文件名列表
    return new_filenames

# 定义一个名为GetData的函数，它接受两个参数：
# data_folder - 数据文件夹路径
# otherinfo - 其他信息字符串，用于构建文件名
def GetData(data_folder, otherinfo):
    # 构建白板参考数据的raw文件路径
    # 使用os.path.join来确保路径在不同操作系统下的兼容性
    # 文件名格式为'WHITEREF_{otherinfo}.raw'
    white_dataPath = os.path.join(data_folder, f'WHITEREF_{otherinfo}.raw')
    # 构建白板参考数据的hdr(头文件)路径
    # 文件名格式为'WHITEREF_{otherinfo}.hdr'
    white_headerPath = os.path.join(data_folder, f'WHITEREF_{otherinfo}.hdr')

    # 构建暗板参考数据的raw文件路径
    # 文件名格式为'DARKREF_{otherinfo}.raw'
    dark_dataPath = os.path.join(data_folder, f'DARKREF_{otherinfo}.raw')
    # 构建暗板参考数据的hdr(头文件)路径
    # 文件名格式为'DARKREF_{otherinfo}.hdr'
    dark_headerPath = os.path.join(data_folder, f'DARKREF_{otherinfo}.hdr')

    # 构建原始数据的raw文件路径
    # 文件名格式为'{otherinfo}.raw'
    raw_dataPath = os.path.join(data_folder, f'{otherinfo}.raw')
    # 构建原始数据的hdr(头文件)路径
    # 文件名格式为'{otherinfo}.hdr'
    raw_headerPath = os.path.join(data_folder, f'{otherinfo}.hdr')

    # 使用GetRawData函数读取原始数据
    # 传入原始数据的头文件路径和数据文件路径      
    rawimg_data = GetRawData(raw_headerPath, raw_dataPath)
    # 使用GetRawData函数读取白板参考数据
    # 传入白板参考数据的头文件路径和数据文件路径
    whiteimg_data = GetRawData(white_headerPath, white_dataPath)
    # 使用GetRawData函数读取暗板参考数据
    # 传入暗板参考数据的头文件路径和数据文件路径
    darkimage_data = GetRawData(dark_headerPath, dark_dataPath)

    # 获取数据的维度信息
    num_samples, num_pixels, num_wavelengths = rawimg_data.shape
    # 获取波长信息
    # 使用之前定义的GetWavelengths函数，传入原始数据的头文件路径
    # 返回一个包含各波段中心波长的数组
    wavelengths = GetWavelengths(raw_headerPath)
    # 根据第一个波长的值判断传感器类型
    # 如果第一个波长大于900nm，则判定为FX17传感器
    # 否则判定为FX10传感器
    if float(wavelengths[0])>900:
        sensors = "FX17"
    else:
        sensors = "FX10"
    # 返回处理结果,包括:
    # - 白板参考数据
    # - 暗板参考数据
    # - 原始数据
    # - 波长信息数组
    # - 传感器类型标识
    return whiteimg_data, darkimage_data, rawimg_data, wavelengths, sensors


def generateMask(rawimg_data,whiteimg_data,darkimage_data, sensors, last_whitedata_means, filename, white_file, last_white_data):
    # 2. 选择特定的波段（目前选择的是第50个波段）从黑白和原始数据中计算出反射率，这个波长下种子与背景反射率最大，因此用来提取mask
    # 根据传感器类型选择不同的波段用于计算mask
    if sensors == "FX17":
        mask_wavenum = 50  # 对于FX17传感器，使用第50个波段
    else:
        mask_wavenum = 410  # 对于其他传感器(如FX10)，使用第410个波段
    # 从三维数据中提取指定波段的二维数据(去除波段维度)
    # np.squeeze用于移除数组中大小为1的维度

    rawdata = np.squeeze(rawimg_data[:,:,mask_wavenum])    # 原始数据指定波段
    whitedata = np.squeeze(whiteimg_data[:,:,mask_wavenum])    # 白板数据指定波段
    darkdata = np.squeeze(darkimage_data[:,:,mask_wavenum])    # 暗板数据指定波段

    # 计算暗板数据的列均值(对每个像素位置在所有样本上的平均值)
    dark_means = np.mean(darkdata, axis=0)
    # 加判断:如果白板标准差std大于200或者矫正后平均反射率数值小于500或者白板平均光谱反射数值小于1000,则跳过该条数据的白板,采用上一个白板数据,并返回该条数据的图像名称
    # 这些条件可能表明白板数据异常
    if np.std(whitedata) >= 200 or np.mean(whitedata) <= 1000:
        # 如果白板数据异常，则使用上一个有效白板数据
        whitedata_means = last_whitedata_means
        white_data = last_white_data
        # 打印警告信息
        print(f"whiteboard data of {filename} is incorrect")

        # 定义备用白板数据的参数
        otherinfo1 = "25-28-2019-1_2024-12-26_02-38-17"
        data_folder1 = "/media/dell/RaspiberryData_556/Hyperspectral/Wheat_Vigor/25-32/petridish/FX17/25-28-2019-1_2024-12-26_02-38-17/capture"
        white_dataPath1 = os.path.join(data_folder1, f'WHITEREF_{otherinfo1}.raw')
        white_headerPath1 = os.path.join(data_folder1, f'WHITEREF_{otherinfo1}.hdr')
        # 重新加载备用白板数据
        whiteimg_data = GetRawData(white_headerPath1, white_dataPath1)
        whitedata = np.squeeze(whiteimg_data[:,:,mask_wavenum])

        # 再次检查备用白板数据是否正常
        if np.std(whitedata) >= 200 or np.mean(whitedata) <= 1000:
            # 如果备用白板数据也不正常，将错误信息写入文件
            with open(white_file, 'a') as f:
                f.write(f"whiteboard data of {data_folder1} is incorrect\n")
        else:
            # 如果备用白板数据正常，计算其均值
            whitedata_means = np.mean(whitedata, axis=0).astype('float64')
            white_data = whiteimg_data

        with open(white_file, 'a') as f:
            f.write(f"whiteboard data of {filename} is incorrect\n")
    else:
        # 如果备用白板数据正常，计算其均值
        whitedata_means = np.mean(whitedata, axis=0).astype('float64')
        white_data = whiteimg_data
    
    # 计算反射率计算的分子部分(白板均值-暗板均值)
    fenzi = whitedata_means - dark_means

    # 计算mask图像(原始反射率归一化到0-10000范围)
    # 这里使用原始数据-暗板均值，然后除以(白板均值-暗板均值)进行归一化
    # 最后乘以10000并四舍五入    
    maskpic = np.round((rawdata - dark_means) / fenzi * 10000)      # 四舍五入的方法保存结果，并保存为16位无符号格式

    # 计算最终的mask图像
    # 将maskpic转换为16位无符号整数格式
    result_pic = maskpic.astype(np.uint16)
    # 使用Otsu方法进行二值化阈值分割
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU表示使用Otsu方法自动确定最佳阈值
    # 结果是一个二值图像(0和255)
    # 对剩下的部分进行阈值分割或边缘检测
    # 这里使用 OpenCV 中的阈值分割作为示例，之后再详细尝试其他方法
    _, mask = cv2.threshold(result_pic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        # 这里得到了种子的mask二值图像用于后面的计算
    # 将mask转换为8位无符号整数格式(节省空间)
    mask = mask.astype(np.uint8)
    # 创建mask的副本
    mask_c = mask.copy()
    # 将mask中的255值替换为1(可能是为了后续计算方便)
    mask[mask == 255] = 1

    # 返回结果:
    # - mask: 二值化后的mask(0和1)
    # - mask_c: 原始二值化mask(0和255)
    # - whitedata_means: 白板数据均值(用于后续计算)
    # - white_data: 使用的白板数据(可能是当前或上一个有效数据)
    return mask, mask_c, whitedata_means, white_data

def labelimage(mask):
    # 找所有的物体，例如有90个种子，则最终label的数值是90，也就是考虑单个种子
    # 使用OpenCV的findContours函数找到mask中的所有轮廓
    # cv2.RETR_EXTERNAL表示只检测最外层轮廓(不检测嵌套轮廓)
    # cv2.CHAIN_APPROX_SIMPLE表示压缩水平、垂直和对角线段，仅保留其端点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 设置面积阈值，用于过滤掉太小的轮廓(可能是噪声)
    area_threshold = 1000

    # 存储每个有效物体的边界框(Bounding Box)信息
    bounding_boxes = []

    # 遍历每个物体的轮廓并获取BBox信息
    for contour in contours:
        # 计算轮廓的边界框(x,y,w,h)，其中(x,y)是左上角坐标，w是宽度，h是高度
        x, y, w, h = cv2.boundingRect(contour)
        # 计算轮廓的面积
        seedarea = cv2.contourArea(contour)
        # 如果轮廓面积大于阈值，则认为是一个有效的种子，保存其边界框信息
        if seedarea > area_threshold:
            bounding_boxes.append((x, y, w, h))
    # print(bounding_boxes)

    # 根据边界框的y坐标(垂直位置)和x坐标(水平位置)对边界框进行排序
    # 先按y坐标排序(从上到下)，y坐标相同的再按x坐标排序(从左到右)
    bounding_boxes.sort(key=lambda box: (box[1], box[0]))  # 先按y坐标，再按x坐标排序
    # 定义组内和组间的距离阈值(像素单位)
    # threshold_within_group: 同一组内种子之间的最大垂直距离
    # threshold_between_group: 不同组之间种子的最小垂直距离
    threshold_within_group = 50
    threshold_between_group = 100
    # 初始化分组相关变量
    group_num = 0   # 组号计数器
    groups = {}     # 存储分组的字典，键是组ID，值是该组的边界框列表
    current_group = []    # 当前正在构建的组

    # 遍历所有边界框，进行分组
    for i in range(len(bounding_boxes)):
        item = bounding_boxes[i]    # 当前边界框

        # 如果是第一个边界框，或者当前边界框与前一个边界框的y坐标差超过组间阈值
        if i == 0 or abs(item[1] - bounding_boxes[i-1][1]) > threshold_between_group:
            # 新建一个组
            group_num += 1
            group_id = str(group_num)   # 组ID(字符串形式)
            current_group = [item]      # 当前组只包含当前边界框
            groups[group_id] = current_group    # 将当前组添加到分组字典中
        else:
            # 如果当前边界框与前一个边界框的y坐标差在组内阈值范围内
            if abs(item[1] - current_group[-1][1]) <= threshold_within_group:
                # 将当前边界框添加到当前组
                current_group.append(item)
            else:
                # 如果当前边界框与当前组最后一个边界框的y坐标差超过组内阈值，新建一个组
                group_num += 1
                group_id = str(group_num)
                current_group = [item]
                groups[group_id] = current_group

    # 对每个组内的边界框按x坐标(从左到右)进行排序
    for group_id in groups:
        groups[group_id] = sorted(groups[group_id], key=lambda x: x[0])

    # 初始化最终排序列表
    sorted_list = []
    # 初始化标签计数器(从1开始)
    counter = 1
    # 创建一个与输入mask大小相同的标签图像，初始化为全零
    labeled_image = np.zeros_like(mask, dtype=np.uint8)
    # 遍历所有分组和组内的边界框
    for group_id, group_items in groups.items():
        for seed_num, seed_info in list(enumerate(group_items)):
            x, y, w, h = seed_info      # 当前边界框信息
            label = counter  # 当前标签值(从1开始递增)
            # 在标签图像中，将当前边界框区域内的mask值乘以标签值
            # 这样每个种子区域会有唯一的标签值
            labeled_image[y:y+h, x:x+w] = mask[y:y+h, x:x+w]*label
            # 创建一个包含种子信息的数组:
            # [边界框信息, 标签值, 组ID, 种子序号(从1开始)]
            circle_with_label = np.append([seed_info], [int(counter), int(group_id), int(seed_num+1)])  # 添加标签
            # 将种子信息添加到最终排序列表中
            sorted_list.append(circle_with_label)
            # 标签计数器递增
            counter += 1

    # 返回结果:
    # - labeled_image: 标签图像，每个种子区域有唯一的标签值
    # - sorted_list: 包含所有种子信息的列表，按从上到下、从左到右的顺序排列
    return labeled_image, sorted_list

# 定义一个名为labelimage_fixed的函数，它接受4个参数：
"""
# mask
# row
# col
# min_area
"""
def labelimage_fixed(mask, row=4, col=2,min_area = 1000):
    # 将输入mask转换为8位无符号整数类型(0-255)
    mask=mask.astype(np.uint8)
    # 获取mask的高度和宽度
    img_h, img_w = mask.shape

    # 初始化存储网格中心位置的列表
    grid_positions = []
    # 按照指定的行数(row)和列数(col)生成网格中心点坐标
    for r in range(row):
        for c in range(col):
            # 计算每个网格的中心x坐标(列方向)
            x = int((c+0.5)*img_w/col)
            # 计算每个网格的中心y坐标(行方向)
            y = int((r+0.5)*img_h/row)
            # 将中心点坐标添加到列表中
            grid_positions.append((x,y))

    # 轮廓提取
    # 使用OpenCV的findContours函数检测mask中的所有轮廓
    # cv2.RETR_EXTERNAL表示只检测最外层轮廓
    # cv2.CHAIN_APPROX_SIMPLE表示压缩轮廓点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 初始化存储有效轮廓中心点坐标的列表
    centers = []

    # 遍历所有检测到的轮廓
    for contour in contours:
        # 只处理面积大于等于min_area(默认1000)的轮廓
        if cv2.contourArea(contour) >= min_area:
            # 计算轮廓的矩(用于计算质心)
            M = cv2.moments(contour)
            # 确保矩不为零(避免除以零错误)
            if M["m00"] != 0 :
                # 计算轮廓的质心x坐标
                cX = int(M["m10"]/M["m00"])
                # 计算轮廓的质心y坐标
                cY = int(M["m01"]/M["m00"])
                # 将质心坐标和轮廓本身添加到列表中
                centers.append((cX,cY,contour))


    # 网格匹配
    # 初始化分配列表，用于存储每个网格位置分配到的轮廓
    # 初始值为None，表示该网格位置尚未分配到轮廓
    assigned = [None]*(row*col)
    # 初始化已使用网格位置的集合
    used = set()

    # 遍历所有有效轮廓的质心
    for cx, cy, contour in centers:
        # 初始化最小距离为无穷大
        min_dist = float('inf')
        # 初始化最小距离对应的网格索引
        min_idx = -1
        # 遍历所有网格位置
        for i, (gx,gy) in enumerate(grid_positions):
            # 跳过已经分配过的网格位置
            if i in used:
                continue
            # 计算当前质心到网格中心的欧氏距离
            dist = math.hypot(cx-gx, cy-gy)
            # 如果距离小于当前最小距离，则更新最小距离和对应的网格索引
            if dist < min_dist:
                min_dist = dist
                min_idx= i
        # 如果找到了合适的网格位置(且距离小于300像素)
        if min_idx != -1 and min_dist < 300:
            # 将当前轮廓分配到该网格位置
            assigned[min_idx] = contour
            # 标记该网格位置为已使用
            used.add(min_idx)
    
    # 生成标签图像
    # 创建一个与输入mask大小相同的标签图像，初始化为全零
    labeled_image = np.zeros_like(mask, dtype=np.uint8)
    # 初始化存储排序后种子信息的列表
    sorted_list = []

    # 遍历所有网格位置及其分配到的轮廓
    for idx, contour in enumerate(assigned):
        # 计算当前种子的标签值(从1开始)
        label = idx + 1
        # 计算当前种子所在的网格行号(从1开始)
        grid_row = idx // col + 1
        # 计算当前种子所在的网格列号(从1开始)
        grid_col = idx % col + 1

        # 如果当前网格位置分配到了轮廓
        if contour is not None:
            # 计算轮廓的边界框(x,y,w,h)
            x, y, w, h = cv2.boundingRect(contour)
            # 在标签图像中，将当前轮廓区域填充为对应的标签值
            cv2.drawContours(labeled_image, [contour], -1, label, thickness=cv2.FILLED)
            # 将种子信息添加到排序列表中
            # 包括:边界框坐标、标签值、网格行号、网格列号
            sorted_list.append([x, y, w, h, label, grid_row, grid_col])
        else:
            # 如果当前网格位置没有分配到轮廓，则添加空信息
            # 但仍然保留网格位置信息(标签值、网格行号、网格列号)
            sorted_list.append([None, None, None, None, label, grid_row, grid_col])

    # 返回结果:
    # - labeled_image: 标签图像，每个种子区域有唯一的标签值
    # - sorted_list: 包含所有种子信息的列表，按网格位置排序
    # - assigned: 每个网格位置分配到的轮廓列表    
    return labeled_image, sorted_list, assigned

# 定义了find_largest_inscribed_circles函数，包含3个参数
"""
# binary_image: 二值化输入图像(前景为255,背景为0)
# min_contour_points: 轮廓的最小点数阈值
# min_area: 轮廓的最小面积阈值
"""
def find_largest_inscribed_circles(binary_image, min_contour_points, min_area):
    # Step 1: 查找图像中的所有轮廓
    # 使用cv2.findContours函数检测二值图像中的轮廓
    # cv2.RETR_EXTERNAL: 只检测最外层轮廓，不检测嵌套轮廓
    # cv2.CHAIN_APPROX_SIMPLE: 压缩轮廓点，只保留端点
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初始化存储检测到的内切圆的列表
    circles = []
    
    # 遍历所有检测到的轮廓
    for contour in contours:
        # 检查轮廓是否满足最小点数要求和最小面积要求
        if len(contour) >= min_contour_points and cv2.contourArea(contour) >= min_area:
            # 创建一个与输入图像大小相同的空白掩码
            mask = np.zeros_like(binary_image)
            # 在掩码上绘制当前轮廓(填充为白色255)
            # -1表示绘制所有轮廓，thickness=cv2.FILLED表示填充轮廓内部
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Step 2: 计算距离变换
            # cv2.distanceTransform计算每个前景像素到最近背景像素的距离
            # cv2.DIST_L2: 使用L2距离度量(欧氏距离)，更精确但计算量稍大
            # 5: 邻域大小，用于计算距离的核大小
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            # Step 3: 找到距离变换中的最大值及其位置
            # cv2.minMaxLoc返回(min_val, max_val, min_loc, max_loc)
            # max_val: 距离变换中的最大值(即最大内切圆的半径)
            # max_loc: 最大值的位置(即圆心坐标)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
            
            # max_loc是最大内切圆的圆心坐标
            center = max_loc
            # max_val是最大内切圆的半径(浮点数)
            # 将半径转换为整数(向下取整)
            radius = int(max_val)
            
            # 将检测到的圆(圆心x,y和半径)添加到结果列表
            circles.append((center[0], center[1],radius))
    
    # 返回所有检测到的内切圆列表
    return circles

# 定义了一个labelCircle的函数，有下列参数：
"""
# mask_c: 二值化掩码图像(用于圆检测)
# resultC: 结果数据(未直接使用，但保留在参数中以保持接口一致)
# filename: 输出文件名(不含扩展名)
# pic_img: 原始图像(用于绘制标记)
# min_contour_points: 轮廓的最小点数阈值(默认100)
# min_area: 轮廓的最小面积阈值(默认5000像素)
# outRawPath: 输出目录路径(可选)
"""
def labelCircle(mask_c, resultC, filename, pic_img, min_contour_points=100, min_area=5000):
    # 假设 mask_c 是你的二值图像
    binary_image = mask_c

    # 反转图像(将前景和背景互换)
    # 这样做的目的是为了使用cv2.findContours的默认行为(检测白色前景轮廓)
    inverted_image = cv2.bitwise_not(binary_image)

    # 找到所有轮廓
    # cv2.RETR_CCOMP: 检索所有轮廓，并将它们组织成两级层次结构
    # cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段，仅保留其端点
    contours, hierarchy = cv2.findContours(inverted_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像，用于绘制填充的轮廓
    # 这个图像将用于后续的圆检测
    filled_image = binary_image.copy()

    # 使用之前定义的函数查找最大内切圆，参数:
    # filled_image: 输入二值图像
    # min_contour_points: 轮廓最小点数阈值
    # min_area: 轮廓最小面积阈值
    circles = find_largest_inscribed_circles(filled_image, min_contour_points, min_area)

    # 按照圆的y坐标(垂直位置)进行排序
    # 这样可以将圆从上到下排列，便于后续分组
    sorted_circles = sorted(circles, key=lambda x: x[1])

    # Grouping circles (将圆分组)
    # 初始化分组相关变量
    groups = []     # 存储所有分组的列表
    current_group = []      # 当前正在构建的分组
    prev_y = None   # 前一个圆的y坐标(初始为None)

    # 遍历所有检测到的圆，按照垂直位置进行分组
    for circle in sorted_circles:
        # 如果是第一个圆或者当前圆的y坐标与前一个圆的y坐标差大于30像素
        if prev_y is None or abs(circle[1] - prev_y) > 30:
            # 如果当前分组不为空，则将其添加到分组列表中
            if current_group:
                groups.append(current_group)
                current_group = []      # 开始新的分组
        # 将当前圆添加到当前分组
        current_group.append(circle)
        # 更新前一个圆的y坐标
        prev_y = circle[1]

    # 如果当前分组不为空(处理最后一组)
    if current_group:
        groups.append(current_group)

        
    # 在每个组内按照x坐标排序并添加标签
    labeled_circles = []    # 存储带有标签的圆信息
    counter = 1             # 全局计数器(从1开始)
    group_num = 1           # 组号计数器(从1开始)

    # 遍历所有分组
    for group in groups:
        # 按照圆的x坐标(水平位置)对组内圆进行排序
        sorted_group = sorted(group, key=lambda x: x[0])
        group_counter = 1  # 在每个组内重新开始编号，组内计数器(从1开始)

        # 遍历当前分组内的所有圆
        for circle in sorted_group:
            # 为圆添加标签信息:
            # [x, y, radius, global_label, group_num, group_in_label]
            # 使用np.append将标签信息添加到圆的信息中
            circle_with_label = np.append(circle, [int(counter), int(group_num), int(group_counter)])  # 添加标签
            # 将带有标签的圆添加到结果列表中
            labeled_circles.append(circle_with_label)
            # 全局计数器递增
            counter += 1
            # 组内计数器递增
            group_counter += 1
        # 组号计数器递增
        group_num += 1

    # 创建一个与输入mask大小相同的标签图像，初始化为全零
    labeled_imC = np.zeros_like(mask_c)
    # 遍历所有带有标签的圆
    for i, circle in enumerate(labeled_circles):
        # 获取圆的参数(圆心x,y,半径,全局编号,组号,组内编号)
        x, y, r, totalnum, group_num, groupin_num = circle
        # 在mask上画圆
        # 绘制一个灰色圆环(半径r)，可能用于可视化检测到的圆
        # 注意: 这里使用r作为半径，而不是r-10(与之前代码不同)
        cv2.circle(mask_c, (x,y), (r), (128, 128, 128), 1)     # 添加线宽参数1
        # 在原始图像pic_img上画圆
        # 绘制一个红色圆环(半径r)，用于在原始图像上标记检测到的圆
        cv2.circle(pic_img, (x,y), (r), (0, 0, 255), 1)        # 添加线宽参数1 
        
        # 根据半径创建一个圆形的mask
        mask_circle = np.zeros_like(mask_c)
        # 在mask_circle上绘制填充的圆形(半径r)
        cv2.circle(mask_circle, (x, y), r, 1, thickness=-1)    # thickness=-1表示填充
        
        # 将mask_circle中的像素值与对应圆的标签值相乘
        # 这样每个圆区域会被赋予对应的标签值
        labeled_imC += mask_circle * totalnum
        
        # 以下代码用于保存标记图像(可选)
        # 注意: outRawPath变量应该在函数外部定义
        # 这部分代码应该放在函数外部调用，而不是在循环内部
        # 否则会导致重复保存图像
        """
        cv2_singleimg = os.path.join(outRawPath, f'{filename}.jpg')
        cv2.imwrite(cv2_singleimg, mask_c)
        cv2_picimg = os.path.join(outRawPath, f'{filename}_raw.jpg')
        cv2.imwrite(cv2_picimg, pic_img)
        """
    if outRawPath is not None:
        # 确保输出目录存在
        os.makedirs(outRawPath, exist_ok=True)
        # 生成输出文件路径
        cv2_singleimg = os.path.join(outRawPath, f'{filename}.jpg')
        cv2_picimg = os.path.join(outRawPath, f'{filename}_raw.jpg')
        # 保存标记后的mask_c图像
        cv2.imwrite(cv2_singleimg, mask_c)
        # 保存标记后的原始图像
        cv2.imwrite(cv2_picimg, pic_img)

    # 返回结果:
    # - labeled_imC: 标签图像，每个种子区域有唯一的标签值
    # - labeled_circles: 包含所有种子信息的列表，按从上到下、从左到右的顺序排列
    return labeled_imC, labeled_circles

# 定义了根据标签图像和光谱数据计算每个区域光谱结果的函数，根据label与光谱数据做乘法得到对应期望的结果。参数有：
# labeled_image: 带标签的图像，每个区域有唯一标签值
# num_values_per_row: 每行包含的标签值数量(用于结果重组)
# wavelengths: 波长列表，对应光谱数据的波长
# rawimg_data: 原始高光谱图像数据
# whiteimg_data: 白板参考图像数据
# darkimage_data: 暗板参考图像数据
def caculateresult(labeled_image,num_values_per_row,wavelengths,rawimg_data,whiteimg_data,darkimage_data):
    # 假设image1是包含不同数值label的图像   
    # 获取标签图像中的最大标签值   
    max_value = labeled_image.max()
    # 初始化结果列表，用于保存每个区域的平均值
    results = []

    # 遍历所有标签值(从1到最大标签值)
    for value in range(1, max_value + 1):
        # 初始化当前标签值的结果列表
        value_results = []
        # 创建二值化阈值图像，当前标签值区域为1，其他区域为0
        # 然后除以标签值(这一步似乎没有实际意义，可能是为了归一化)
        threshold_image = np.where(labeled_image == value, labeled_image, 0) / value
        # 计算当前标签区域的像素数量(面积)
        area_pixels = np.count_nonzero(threshold_image)

        # 如果区域面积大于0，则计算该区域的光谱平均值
        if area_pixels > 0:
            for i, wavelength in enumerate(wavelengths):
                # 从原始图像数据中提取当前波段的数据
                rawdata = np.squeeze(rawimg_data[:,:,i])
                # 从白板参考数据中提取当前波段的数据
                whitedata = np.squeeze(whiteimg_data[:,:,i])
                # 从暗板参考数据中提取当前波段的数据
                darkdata = np.squeeze(darkimage_data[:,:,i])
                
                # 计算白板数据的均值(对每个像素位置在所有样本上的平均值)
                whitedata_means = np.mean(whitedata, axis=0).astype('float64')
                # 计算暗板数据的均值
                dark_means = np.mean(darkdata, axis=0)
                # 计算反射率计算的分母部分(白板均值-暗板均值)
                fenzi = whitedata_means - dark_means
                # 计算反射率图像(原始反射率归一化到0-10000范围)
                # 公式: (rawdata - dark_means) / fenzi * 10000
                result_pic = np.round((rawdata - dark_means) / fenzi * 10000)
                # 将负值设为0，并转换为16位无符号整数格式
                spectral_img = np.where(result_pic < 0, 0, result_pic).astype(np.uint16)    
                # 计算当前波段在当前标签区域的反射率总和        
                area_sum = np.sum(spectral_img * threshold_image)
                # 计算当前波段在当前标签区域的反射率平均值
                area_mean = area_sum / area_pixels
                # 将计算结果添加到当前标签值的结果列表中
                results.append(area_mean)
    # 计算结果的行数(用于结果重组)
    # 假设每个标签值对应num_values_per_row个波段结果
    num_rows = int(len(results) / num_values_per_row)
    # 返回计算结果和行数
    return results, num_rows
"""
# 定义了writeresult函数，将结果写入CSV文件。包含参数：
# output_csv: 输出CSV文件路径
# num_rows: 结果的行数(用于结果重组)
# results: 计算得到的光谱结果数组(形状: [标签值数量, 波段数量])
# header_written: 布尔值，指示是否已写入表头
# wavelengths: 波长列表，对应光谱数据的波长
# num_values_per_row: 每行包含的标签值数量(用于结果重组)
# prefix: 文件名前缀列表(用于生成CSV中的文件名列)
# pic_scale: 图像尺寸标识(未在函数中使用)
# sortedC_info: 排序后的圆信息(未在函数中使用)
"""
def writeresult(output_csv, num_rows, results, header_written, wavelengths, num_values_per_row, prefix,  pic_scale, sortedC_info):
    # 使用追加模式打开CSV文件
    with open(output_csv, 'a', newline='') as csvfile:
        # 创建CSV写入器对象
        writer = csv.writer(csvfile)
        # 如果文件不存在或表头尚未写入，则写入表头
        if  header_written == False:
            # 创建表头行: 文件名 + 所有波长
            header_row = ['filename'] + wavelengths
            writer.writerow(header_row)
            # 更新表头写入状态为True
            header_written = True

        # 遍历所有行(每行包含num_values_per_row个标签值的结果)
        for i in range(num_rows):
            # 计算当前行的起始索引
            start_index = i * num_values_per_row
            # 计算当前行的结束索引
            end_index = (i + 1) * num_values_per_row

            # 从prefix列表中获取当前行的文件名
            # 注意: 这里假设prefix列表有足够的元素，否则可能引发IndexError
            filename = prefix[i]
            # 从结果数组中提取当前行的数据
            # 注意: 这里假设results是一维数组，且长度是num_rows * num_values_per_row
            # 但根据前面的函数，results实际上是二维数组[标签值数量, 波段数量]
            # 这里存在参数不匹配的问题
            row_values = [filename] + results[start_index:end_index]

            # 将当前行数据写入CSV文件
            writer.writerow(row_values)

    # 返回更新后的表头写入状态    
    return header_written

# 定义了主处理函数，遍历文件夹中的样品数据并进行处理，主要参数有：
"""
    totalfolder: 包含所有样品文件夹的根目录
    outpath: 输出结果的根目录
    pic_scale: 图像尺寸标识("large"或"small")
    output_csv: 输出CSV文件路径(常规结果)
    output_csvC: 输出CSV文件路径(圆检测结果)
    incorrect_file: 记录处理失败样品的文件路径
    outMaskPath: 输出掩码图像的目录
    outRawPath: 输出原始标记图像的目录
    overnum: 跳过处理的起始编号(可选)
    overrepeat: 跳过处理的重复次数阈值(可选)
"""
def main(totalfolder, outpath, pic_scale, output_csv, output_csvC, incorrect_file, outMaskPath, outRawPath, overnum, overrepeat):
    # 获取所有子文件夹列表并按名称排序
    folder_list = [folder for folder in os.listdir(totalfolder) if os.path.isdir(os.path.join(totalfolder, folder))]
    file_list = sorted(folder_list)
    # 初始化表头写入状态
    header_written = False
    header_writtenC = False
    # 初始化白板数据参考(用于异常检测)
    last_whitedata_means = None
    last_white_data = None
    # 获取总样品数量
    total_samples = len(file_list)

    # 遍历所有样品文件夹
    for i, filename in enumerate(file_list):
        filename = "18-25-1.1_2025-05-20_09-16-29"
        # 记录开始时间
        start_time = time.time()
        # 获取当前时间并格式化
        now = datetime.now().strftime("%H:%M:%S")
        # 打印进度信息
        print(f"\n[{now}] 🟡 正在处理第 {i+1}/{total_samples} 个样品：{filename}")

        # 构建数据文件夹路径
        data_folder = os.path.join(totalfolder, filename, "capture")
        # 解析原始样品文件名获取起始编号、结束编号和重复次数。根据文件名，调整parts[n]
        parts = filename.split('_')
        if len(parts) == 3:
            start, end, repeat = parts[0].split('-')    # 根据不同文件名，调整取parts哪部分的值以及分割符号
        elif len(parts) == 4:
            start, end, repeat = parts[1].split('-')
        else:
            # 文件名格式无法解析，记录错误并跳过
            print(f"❌ 无法解析的文件名格式: {filename}")
            continue # 跳过错误，继续运行代码
        
        # 检查是否需要跳过当前样品(基于overnum和overrepeat参数)
        if overnum is not None:
            if int(start) <= int(overnum):
                # 已经处理过起始编号之前的样品，标记表头已写入
                header_written = True
                header_writtenC = True
                if int(end) < int(overnum):
                    # 当前样品完全在跳过范围内，直接跳过
                    continue
                elif int(overnum) <= int(end):
                    # 当前样品部分在跳过范围内，检查重复次数
                    if float(repeat) <= float(overrepeat):
                        # 重复次数也在跳过范围内，跳过当前样品
                        continue
        # 处理文件名获取前缀信息
        result = process_filename(repeat, start, end)
        if result is None or len(result) > 8:
            # 文件名非法或前缀数量过多，记录错误并跳过
            print(f"❌ 文件名非法或数量过多，跳过样品：{filename}")
            with open(incorrect_file, 'a') as f:
                f.write(f"{filename} \n")
            continue
        # 获取有效的前缀列表
        prefix = result

        # 获取各种图像数据(原始图像、白板图像、暗板图像等)
        whiteimg_data, darkimage_data, rawimg_data, wavelengths, sensors = GetData(data_folder, filename)
        # 计算每个波长对应的值数量(用于结果重组)
        num_values_per_row = len(wavelengths)

        # 生成掩码图像(用于区域分割)
        mask, mask_c, last_whitedata_means, white_data = generateMask(
            rawimg_data, whiteimg_data, darkimage_data, sensors,
            last_whitedata_means, filename, incorrect_file, last_white_data
        )

        # 检查白板数据是否异常
        if not np.array_equal(white_data, whiteimg_data):
            print("⚠️ 白板异常，已使用备用白板")
        else:
            print("✅ 白板有效")

        # 使用固定网格方法标记图像中的区域
        label_img, sorted_info, used = labelimage_fixed(mask)
        # 获取标记的最大值(即区域数量)
        label_count = label_img.max()
        if label_count > 8:
            # 区域数量过多，可能影响处理结果，记录错误并跳过
            print(f"❌ Label 数量过多（{label_count}个），跳过样品")
            with open(incorrect_file, 'a') as f:
                f.write(f"{filename} \n")
            continue
        else:
            # 区域数量正常，打印信息
            print(f"✅ Label 数量：{label_count} 个")

        # 获取实际使用的前缀列表(与标记区域对应的文件名前缀)
        used_pre = [prefix[i] for i, item in enumerate(used) if item is not None]
        # 计算常规结果(基于标记区域)
        pic_result, num_rows = caculateresult(label_img, num_values_per_row, wavelengths, rawimg_data, white_data, darkimage_data)
        # 将结果写入CSV文件
        header_written = writeresult(output_csv, num_rows, pic_result, header_written, wavelengths, num_values_per_row, used_pre, pic_scale, sorted_info)
        # 读取原始图像用于圆检测标记
        pic_path = os.path.join(totalfolder, filename, f"{filename}.png")
        pic_img = cv2.imread(pic_path)
        # 保存掩码图像
        cv2_maskimg = os.path.join(outMaskPath, f'{filename}.jpg')
        cv2.imwrite(cv2_maskimg, mask_c)
        # 使用圆检测方法标记图像中的区域
        labeled_imC, sortedC_info = labelCircle(mask_c, result, filename, pic_img)
        # 计算圆检测结果
        pic_resultC, num_rowsC = caculateresult(labeled_imC, num_values_per_row, wavelengths, rawimg_data, white_data, darkimage_data)
        # 将圆检测结果写入CSV文件
        header_writtenC = writeresult(output_csvC, num_rowsC, pic_resultC, header_writtenC, wavelengths, num_values_per_row, used_pre, pic_scale, sortedC_info)
        # 更新白板数据参考(用于下一样品的白板异常检测)
        last_white_data = white_data

        # 计算并打印处理用时
        elapsed = time.time() - start_time
        print(f"⏱️ 用时：{elapsed:.2f} 秒")
        print("🟩 样品处理完成")
        
if __name__ == "__main__":
    # 记录程序开始时间和当前时间
    t1 = time.time()
    current_time = datetime.now().strftime("%Y%m%d")
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 原始数据总目录
    # raw_root = Path(r'Z:/00.rawdata/')
    # 调试目录
    raw_root = Path(r'Z:/00.rawdata/debug/')
    # 输出结果根目录
    result_root = Path(r'Z:/03.result/') 

    # 遍历所有子文件夹
    for folder in raw_root.iterdir():
        if folder.is_dir():
            # 如果文件夹名包含 "2pics"，则跳过
            if '2pics' in folder.name:
                print(f"跳过文件夹（包含 '2pics'): {folder.name}")
                continue
        
        totalfolder = folder  #  保留变量名不变
        outname = f"{totalfolder.name}_result_{current_time}"  # 输出文件夹名称
        outpath = os.path.join(result_root, outname)  # 输出完整路径

        Path(outpath).mkdir(parents=True, exist_ok=True)

        print(f"正在处理: {totalfolder}")
        print(f"输出路径: {outpath}")

    # 创建输出目录(如果不存在)
    os.makedirs(outpath, exist_ok=True)
    pic_scale = "large"      # 图像尺寸标识
    output_csv = os.path.join(outpath, f'{outname}.csv')    # 常规结果CSV路径
    output_csvC = os.path.join(outpath, f'{outname}_C.csv')     # 圆检测结果CSV路径

    # 检查输出CSV是否已存在(用于断点续处理)
    if Path(output_csv).exists():
        # 读取已处理的最后一个文件名
        df = pd.read_csv(output_csv, sep=",")
        overfilename = list(df["filename"])[-1]
        # 解析起始编号和重复次数
        overnum, overrepeat = overfilename.split('-')
    else:
        # 首次运行，没有需要跳过的样品
        overnum = None
        overrepeat = None

    # 定义错误记录文件路径
    incorrect_file = os.path.join(outpath, f'incorrect.txt') 
    # 定义掩码图像输出目录
    outMaskPath = os.path.join(outpath, 'mask')
    # 定义原始标记图像输出目录
    outRawPath = os.path.join(outpath, 'Raw')
    # 创建输出目录(如果不存在)
    os.makedirs(outMaskPath, exist_ok=True)
    os.makedirs(outRawPath, exist_ok=True)

    # 调用主处理函数
    main(totalfolder, outpath, pic_scale, output_csv, output_csvC, incorrect_file, outMaskPath, outRawPath, overnum, overrepeat)
    # 记录程序结束时间并计算总用时
    t2 = time.time()
    print("{:.4f}h".format((t2 - t1) / (60 * 60)))    # 转换为小时并打印