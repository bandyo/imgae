#### 这段代码是为了处理排列比较整齐的种子图像
### 首先读取数据
#### 然后选择一个波段用来做mask的提取，分割种子和背景
### 其次根据mask中各个种子的位置排序
#### 最后将每个种子乘以对应区域的光谱数据，得到各个波长下的光谱反射率数值

### 这个文件是为了处理排列比较整齐的种子图像
import cv2
import numpy as np
from glob import glob
from pathlib import Path
import exifread
import spectral.io.envi as envi
import csv
import matplotlib.pyplot as plt
import os
import tifffile
import time
import pandas as pd


### 1读取数据
def GetWavelengths(headerPath):
    header = envi.read_envi_header(headerPath)
    wavelengths = header['wavelength']
    return wavelengths
def GetRawData(headerPath,dataPath):
    image_data=envi.open(headerPath,dataPath)
    image_data=image_data.load()
    return image_data
 
    
def process_filename(repeat, start ,end):### 这里针对的是gz_001-008-1.1_2025-06-15_01-57-11这样的文件名


    # position_type = int(parts[2])      # 位置排列方式，1或2
    position_type = int(repeat.split('.')[0] if '.' in repeat else 1 ) # 位置排列方式，1或2
    # repeat = int(filename.split('.')[1]) if '.' in filename else 1  # 重复次数


    # 生成基础数字序列
    start = int(start)
    # end = int(end)
    # numbers = [start, end]
    # 根据位置排列方式调整顺序
    numbers = [1,2]
    if position_type == 1:
        arranged = [1,2]
    elif position_type == 2:
        arranged = [2,1]
        # arranged = []
        # for i in range(0, len(numbers), 2):
        #     if i+1 < len(numbers):
        #         arranged.extend([numbers[i+1], numbers[i]])
        #     else:
        #         arranged.append(numbers[i])
    new_filenames = [f"{start}-{position_type}.{i}" for i in arranged]
    # new_filenames = [1,2,3,4]

    return new_filenames


def GetData(data_folder, otherinfo):
    ##White board
    white_dataPath = os.path.join(data_folder, f'WHITEREF_{otherinfo}.raw')
    white_headerPath = os.path.join(data_folder, f'WHITEREF_{otherinfo}.hdr')

    ##Dark board
    dark_dataPath = os.path.join(data_folder, f'DARKREF_{otherinfo}.raw')
    dark_headerPath = os.path.join(data_folder, f'DARKREF_{otherinfo}.hdr')

    ##Raw data
    raw_dataPath = os.path.join(data_folder, f'{otherinfo}.raw')
    raw_headerPath = os.path.join(data_folder, f'{otherinfo}.hdr')
          
    rawimg_data = GetRawData(raw_headerPath, raw_dataPath)
    whiteimg_data = GetRawData(white_headerPath, white_dataPath)
    darkimage_data = GetRawData(dark_headerPath, dark_dataPath)
    # reflectimage_data = GetRawData(Reflect_headerPath, Reflect_dataPath)


    # ### 获取数据的维度信息
    num_samples, num_pixels, num_wavelengths = rawimg_data.shape
    # 获取波长信息
    wavelengths = GetWavelengths(raw_headerPath)
    if float(wavelengths[0])>900:
        sensors = "FX17"
    else:
        sensors = "FX10"
    return whiteimg_data, darkimage_data, rawimg_data, wavelengths, sensors


def generateMask(rawimg_data,whiteimg_data,darkimage_data, sensors, last_whitedata_means, filename, white_file, last_white_data):
    ### 2. 选择特定的波段（目前选择的是第50个波段）从黑白和原始数据中计算出反射率，这个波长下种子与背景反射率最大，因此用来提取mask
    ### Select a specific band (currently the 50th band) to compute reflectance and extract the mask
    # start_row = None
    # end_row = None
    if sensors == "FX17":
        mask_wavenum = 50
    else:
        mask_wavenum = 410
    rawdata = np.squeeze(rawimg_data[:,:,mask_wavenum])
    whitedata = np.squeeze(whiteimg_data[:,:,mask_wavenum])
    darkdata = np.squeeze(darkimage_data[:,:,mask_wavenum])

    dark_means = np.mean(darkdata, axis=0)
    ### 加判断:如果白板标准差std大于200或者矫正后平均反射率数值小于500或者白板平均光谱反射数值小于1000,则跳过该条数据的白板,采用上一个白板数据,并返回该条数据的图像名称
    if np.std(whitedata) >= 200 or np.mean(whitedata) <= 1000:
        whitedata_means = last_whitedata_means
        white_data = last_white_data
        print(f"whiteboard data of {filename} is incorrect")
        otherinfo1 = "25-28-2019-1_2024-12-26_02-38-17"
        data_folder1 = "/media/dell/RaspiberryData_556/Hyperspectral/Wheat_Vigor/25-32/petridish/FX17/25-28-2019-1_2024-12-26_02-38-17/capture"
        white_dataPath1 = os.path.join(data_folder1, f'WHITEREF_{otherinfo1}.raw')
        white_headerPath1 = os.path.join(data_folder1, f'WHITEREF_{otherinfo1}.hdr')
        # white_headerPath1=
        whiteimg_data = GetRawData(white_headerPath1, white_dataPath1)
        whitedata = np.squeeze(whiteimg_data[:,:,mask_wavenum])
        if np.std(whitedata) >= 200 or np.mean(whitedata) <= 1000:
            with open(white_file, 'a') as f:
                f.write(f"whiteboard data of {data_folder1} is incorrect\n")
        else:
            whitedata_means = np.mean(whitedata, axis=0).astype('float64')
            white_data = whiteimg_data

        with open(white_file, 'a') as f:
            f.write(f"whiteboard data of {filename} is incorrect\n")
    else:
        whitedata_means = np.mean(whitedata, axis=0).astype('float64')
        white_data = whiteimg_data
    fenzi = whitedata_means - dark_means
    # result_pic = (rawdata - dark_means) / (whitedata_means - dark_means) ###四舍五入的方法保存结果，并保存为16位无符号格式
    maskpic = np.round((rawdata - dark_means) / fenzi * 10000)###四舍五入的方法保存结果，并保存为16位无符号格式

    ### 计算mask,不区分品种时得到前景和背景就可以
    result_pic = maskpic.astype(np.uint16)
    # 对剩下的部分进行阈值分割或边缘检测
    # 这里使用 OpenCV 中的阈值分割作为示例，之后再详细尝试其他方法
    _, mask = cv2.threshold(result_pic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#### 这里得到了种子的mask二值图像用于后面的计算
    mask = mask.astype(np.uint8)
    mask_c = mask.copy()
    mask[mask == 255] = 1
    # mask = mask.astype(np.uint8)
    
    return mask, mask_c, whitedata_means, white_data

def labelimage(mask):
    ### 3.1 找所有的物体，例如有90个种子，则最终label的数值是90，也就是考虑单个种子
    # 找到每个物体的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_threshold = 1000

    # 存储每个物体的BBox信息
    bounding_boxes = []

    # 遍历每个物体的轮廓并获取BBox信息
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        seedarea = cv2.contourArea(contour)
        if seedarea > area_threshold:
            bounding_boxes.append((x, y, w, h))
    # print(bounding_boxes)

    # 根据BBox的位置进行排序
    bounding_boxes.sort(key=lambda box: (box[1], box[0]))  # 先按y坐标，再按x坐标排序
    # sorted_final_list = sorted(bounding_boxes, key=lambda x:x[0])
    threshold_within_group = 50
    threshold_between_group = 100

    group_num = 0
    groups = {}
    current_group = []

    for i in range(len(bounding_boxes)):
        item = bounding_boxes[i]

        if i == 0 or abs(item[1] - bounding_boxes[i-1][1]) > threshold_between_group:
            # 如果当前元素与前一个元素的差距超过了阈值，新建一组
            group_num += 1
            group_id = str(group_num)
            current_group = [item]
            groups[group_id] = current_group
        else:
            # 如果当前元素与前一个元素的差距在阈值范围内，将其添加到当前组
            if abs(item[1] - current_group[-1][1]) <= threshold_within_group:
                current_group.append(item)
            else:
            # 如果当前元素与当前组最后一个元素的差距超过了阈值，新建一组
                group_num += 1
                group_id = str(group_num)
                current_group = [item]
                groups[group_id] = current_group

    for group_id in groups:
        groups[group_id] = sorted(groups[group_id], key=lambda x: x[0])

    sorted_list = []

    counter = 1
    # 创建一个标签图像，初始化为全零
    labeled_image = np.zeros_like(mask, dtype=np.uint8)
    for group_id, group_items in groups.items():
        for seed_num, seed_info in list(enumerate(group_items)):
            x, y, w, h = seed_info
            label = counter  # 标签从1开始
            # label = (int(group_id)-1)*1 + seed_num + 1  # 标签从1开始
            labeled_image[y:y+h, x:x+w] = mask[y:y+h, x:x+w]*label
            circle_with_label = np.append([seed_info], [int(counter), int(group_id), int(seed_num+1)])  # 添加标签
            sorted_list.append(circle_with_label)
            counter += 1

    return labeled_image, sorted_list

def labelC(mask, mask_c, result, filename, pic_scale, sensors, pic_img):
    ####首先找圆，其次生成从上到下的带有label的图像
    mask_c = mask_c.astype(np.uint8)
    if pic_scale == "large":###大培养皿的尺寸
        if sensors == "FX17":
            minR = 200
            maxR = 250
        else:
            minR = 300
            maxR = 400

    else:###小培养皿的尺寸
        if sensors == "FX17":
            minR = 50
            maxR = 100
        else:
            minR = 100
            maxR = 150

    minDistance = 2 * maxR

    # 使用霍夫圆变换检测圆
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist = minDistance, param1 = 1, param2 = 1, minRadius = minR, maxRadius = maxR)

    circles = circles[0]
    # 按照圆心的y坐标排序
    sorted_data = circles[circles[:, 1].argsort()]

    # 分组
    groups = []
    current_group = []
    prev_y = None

    for circle in sorted_data:
        if prev_y is None or abs(circle[1] - prev_y) > minDistance:
            if current_group:
                groups.append(np.array(current_group))
                current_group = []
        current_group.append(circle)
        prev_y = circle[1]

    if current_group:
        groups.append(np.array(current_group))

    # 在每个组内按照x坐标排序并添加标签
    labeled_circles = []
    counter = 1

    # 在每个组内按照x坐标排序并添加标签
    labeled_circles = []
    counter = 1
    group_num = 1
    for group in groups:
        sorted_group = group[group[:, 0].argsort()]
        group_counter = 1  # 在每个组内重新开始编号
        for circle in sorted_group:
            circle_with_label = np.append(circle, [int(counter), int(group_num), int(group_counter)])  # 添加标签
            labeled_circles.append(circle_with_label)
            counter += 1
            group_counter += 1
        group_num += 1
        
    ## 绘制检测到的圆
    # 创建一个空白图像，用于绘制标签
    labeled_imC = np.zeros_like(mask_c)
    sorted_circles = np.uint16(np.around(labeled_circles))
    for i, circle in enumerate(sorted_circles):
        # 获取圆心和半径
        x, y, r, totalnum, group_num, groupin_num = circle.astype(int)
        ### 在mask上画圆
        cv2.circle(mask_c, (x,y), (r-10), (128, 128, 128))
        ### 在原始图像上画圆
        cv2.circle(pic_img, (x,y), (r-10), (0, 0, 255))

        
        # 根据半径创建一个圆形的mask
        mask_circle = np.zeros_like(mask_c)
        # cv2.circle(mask_circle, (x, y), r, 1, thickness=-1)
        
        # 将mask_circle中的像素值与对应圆的标签值相乘
        labeled_imC += mask_circle * totalnum
        
        # filenum = '_'.join(map(str, result[0]))
        cv2_singleimg = os.path.join(outpath, f'{filename}.jpg')
        # cv2_singleimg = os.path.join(outpath, f'{filenum}_{result[1]}.jpg')
        cv2.imwrite(cv2_singleimg, mask_c)
        cv2_picimg = os.path.join(outpath, f'{filename}_raw.jpg')
        cv2.imwrite(cv2_picimg, pic_img)

    return labeled_imC, sorted_circles

def find_largest_inscribed_circles(binary_image, min_contour_points, min_area):
    # Step 1: Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    
    for contour in contours:
        # Filter contours by number of points and area
        if len(contour) >= min_contour_points and cv2.contourArea(contour) >= min_area:
            # Create a mask for the current contour
            mask = np.zeros_like(binary_image)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Step 2: Create a distance transform
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            
            # Step 3: Find the point with the maximum distance to the nearest zero pixel
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
            
            # max_loc is the center of the largest inscribed circle
            # max_val is the radius of the largest inscribed circle
            center = max_loc
            radius = int(max_val)
            
            circles.append((center[0], center[1],radius))
    
    return circles

def labelCircle(mask_c, resultC, filename, pic_img, min_contour_points=100, min_area=5000):###find_largest_inscribed_circles
    # 假设 mask_c 是你的二值图像
    binary_image = mask_c

    # 反转图像
    inverted_image = cv2.bitwise_not(binary_image)

    # 找到所有轮廓
    contours, hierarchy = cv2.findContours(inverted_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像，用于绘制填充的轮廓
    filled_image = binary_image.copy()

    
    circles = find_largest_inscribed_circles(filled_image, min_contour_points, min_area)

    sorted_circles = sorted(circles, key=lambda x: x[1])

    # Grouping circles
    groups = []
    current_group = []
    prev_y = None

    for circle in sorted_circles:
        if prev_y is None or abs(circle[1] - prev_y) > 30:
            if current_group:
                groups.append(current_group)
                current_group = []
        current_group.append(circle)
        prev_y = circle[1]

    if current_group:
        groups.append(current_group)

        
    # 在每个组内按照x坐标排序并添加标签
    labeled_circles = []
    counter = 1
    group_num = 1
    for group in groups:
        # sorted_group = group[group[:, 0].argsort()]
        sorted_group = sorted(group, key=lambda x: x[0])
        group_counter = 1  # 在每个组内重新开始编号
        for circle in sorted_group:
            circle_with_label = np.append(circle, [int(counter), int(group_num), int(group_counter)])  # 添加标签
            labeled_circles.append(circle_with_label)
            counter += 1
            group_counter += 1
        group_num += 1

    labeled_imC = np.zeros_like(mask_c)
    # sorted_circles = np.uint16(np.around(labeled_circles))
    for i, circle in enumerate(labeled_circles):
        # 获取圆心和半径
        x, y, r, totalnum, group_num, groupin_num = circle
        # ### 在mask上画圆
        cv2.circle(mask_c, (x,y), (r), (128, 128, 128))
        ### 在原始图像上画圆
        cv2.circle(pic_img, (x,y), (r), (0, 0, 255))
        
        # 根据半径创建一个圆形的mask
        mask_circle = np.zeros_like(mask_c)
        cv2.circle(mask_circle, (x, y), r, 1, thickness=-1)
        
        # 将mask_circle中的像素值与对应圆的标签值相乘
        labeled_imC += mask_circle * totalnum
        
        cv2_singleimg = os.path.join(outRawPath, f'{filename}.jpg')
        cv2.imwrite(cv2_singleimg, mask_c)
        cv2_picimg = os.path.join(outRawPath, f'{filename}_raw.jpg')
        cv2.imwrite(cv2_picimg, pic_img)
    return labeled_imC, labeled_circles

def caculateresult(labeled_image,num_values_per_row,wavelengths,rawimg_data,whiteimg_data,darkimage_data):
    ### 4 根据label与光谱数据做乘法得到对应期望的结果  
    # 假设image1是包含不同数值label的图像      
    max_value = labeled_image.max()
    # 初始化结果列表，用于保存每个区域的平均值
    results = []

    for value in range(1, max_value + 1):
    # Initialize results for this value
        value_results = []
        # Create threshold image for this value
        threshold_image = np.where(labeled_image == value, labeled_image, 0) / value
        # Calculate area mean for this wavelength and value
        area_pixels = np.count_nonzero(threshold_image)
        if area_pixels > 0:
            for i, wavelength in enumerate(wavelengths):
                rawdata = np.squeeze(rawimg_data[:,:,i])
                whitedata = np.squeeze(whiteimg_data[:,:,i])
                darkdata = np.squeeze(darkimage_data[:,:,i])

                whitedata_means = np.mean(whitedata, axis=0).astype('float64')
                dark_means = np.mean(darkdata, axis=0)
                fenzi = whitedata_means - dark_means
                result_pic = np.round((rawdata - dark_means) / fenzi * 10000)

                spectral_img = np.where(result_pic < 0, 0, result_pic).astype(np.uint16)            
                area_sum = np.sum(spectral_img * threshold_image)
                area_mean = area_sum / area_pixels
                results.append(area_mean)

        # Add results for this value to overall results
        # results.append(value_results)

    num_rows = int(len(results) / num_values_per_row)
    return results, num_rows



def writeresult(output_csv, num_rows, results, header_written, wavelengths, num_values_per_row, prefix,  pic_scale, sortedC_info):
    # 将结果写入CSV文件

    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 如果文件不存在，写入表头
        if  header_written == False:
            header_row = ['filename'] + wavelengths
            writer.writerow(header_row)
            header_written = True
     
        for i in range(num_rows):
            start_index = i * num_values_per_row
            end_index = (i + 1) * num_values_per_row

            filename = prefix[i]
            row_values = [filename] + results[start_index:end_index]
            writer.writerow(row_values)
        
    return header_written


def main(totalfolder,outpath, pic_scale, output_csv, output_csvC, incorrect_file, outMaskPath, outRawPath, overnum, overrepeat):
    folder_list = [folder for folder in os.listdir(totalfolder) if os.path.isdir(os.path.join(totalfolder, folder))]
    # file_list = sorted(folder_list)[253:] # 可以手动设置从253号开始
    file_list = sorted(folder_list)[:1]

    header_written = False
    header_writtenC = False
    last_whitedata_means = None  # 初始化为 None 或者适当的初始值
    last_white_data = None
    for filename in file_list:
        filename = "gz_257-257-2.1_2025-06-15_11-16-33"
        data_folder=os.path.join(totalfolder, filename, "capture")
        
        
        parts = filename.split('_')  # 对文件名以"_"进行分割成N部分
        print(parts) # 打印出分割后的parts结果，可用于提示当前处理的样品名
        start, end, repeat = parts[1].split('-')  # 提取开始和结束数字
        if overnum != None:
            if int(start) <= int(overnum):
                header_written = True
                header_writtenC = True
                if int(end) < int(overnum):
                    continue
                elif int(overnum) <= int(end):
                    if float(repeat) <= float(overrepeat) :
                        continue

        result = process_filename(repeat, start, end) ### 生成品种编号和重复数
        if result == None:
            continue
        if len(result) > 2:
            print(f"image name of {filename} is incorrect")
            with open(incorrect_file, 'a') as f:
                f.write(f"{filename} \n")
            continue
        if result is not None:
            prefix = result

        whiteimg_data, darkimage_data, rawimg_data, wavelengths, sensors = GetData(data_folder, filename)
        num_values_per_row = len(wavelengths)  # 每行的值的数量

        ## 生成种子的mask图像,这一步增加了对白数据的判断,如果有问题则用上一张的数据 Generate seed mask image
        mask, mask_c, last_whitedata_means, white_data = generateMask(rawimg_data,whiteimg_data,darkimage_data, sensors, last_whitedata_means, filename, incorrect_file, last_white_data)

        ## 生成mask对应的label Generate seed label image according to seed mask
        label_img, sorted_info = labelimage(mask)
        if label_img.max() >8 :
            print(f"label image of {filename} is incorrect")
            with open(incorrect_file, 'a') as f:
                f.write(f"{filename} \n")
            continue
        ### 根据mask计算结果
        pic_result, num_rows = caculateresult(label_img,num_values_per_row,wavelengths,rawimg_data,white_data,darkimage_data)
        # result_out = writeresult(output_csv,num_rows, pic_result, header_written)
        header_written = writeresult(output_csv,num_rows, pic_result, header_written, wavelengths, num_values_per_row, prefix, pic_scale, sorted_info)
        
        pic_path = os.path.join(totalfolder,filename,f"{filename}.png")
        pic_img = cv2.imread(pic_path)
        cv2_maskimg = os.path.join(outMaskPath, f'{filename}.jpg')
        cv2.imwrite(cv2_maskimg, mask_c)

        ### 根据圆计算结果
        ## 生成圆对应的label Generate circle label image 

        labeled_imC, sortedC_info = labelC(mask, mask_c,result, filename, pic_scale, sensors, pic_img) ## 根据半径找圆
        labeled_imC, sortedC_info = labelCircle(mask_c,result, filename,pic_img)## 根据二值图轮廓找最大内接圆
        pic_resultC, num_rowsC = caculateresult(labeled_imC, num_values_per_row, wavelengths, rawimg_data, white_data, darkimage_data)
        # result_out = writeresult(output_csv,num_rows, pic_result, header_written)
        header_writtenC = writeresult(output_csvC, num_rowsC, pic_resultC, header_writtenC, wavelengths, num_values_per_row, prefix, pic_scale, sortedC_info)
        last_white_data = white_data

if __name__ == "__main__":
    t1 = time.time()
    
    totalfolder = Path(r'E:/HYPERSPECTRAL/guizhou_Zhou/FX10_GZ')
    outname = "FX17_SAP"
    outpath = "E:/HYPERSPECTRAL/guizhou_Zhou/FX17_SAP_result1"

    outpath = os.path.join(outpath, outname)

    os.makedirs(outpath,exist_ok=True)
    # pic_scale = "small"
    pic_scale = "large"
    output_csv = os.path.join(outpath, f'{outname}.csv') 
    output_csvC = os.path.join(outpath, f'{outname}_C.csv') 
    if Path(output_csv).exists():
        df =  pd.read_csv(output_csv, sep=",")
        overfilename = list(df["filename"])[-1]
        overnum, overrepeat = overfilename.split('-')
    else:
        overnum = None
        overrepeat = None

    incorrect_file = os.path.join(outpath, f'incorrect.txt') 

    outMaskPath = os.path.join(outpath, 'mask')
    outRawPath = os.path.join(outpath, 'Raw')
    os.makedirs(outMaskPath,exist_ok=True)
    os.makedirs(outRawPath,exist_ok=True)

    main(totalfolder,outpath, pic_scale, output_csv, output_csvC, incorrect_file,outMaskPath, outRawPath, overnum, overrepeat)
    t2 = time.time()
    print("{:.4f}h".format((t2 - t1) / (60 * 60)))




