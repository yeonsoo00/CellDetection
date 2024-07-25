from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import json
import os
import pandas as pd 

"""
TODO
    - Write def sum_intensity 
    - How to deal with a cell border?
    - Make a loop for all of genes
"""

def is_circle(stats, aspect_ratio_threshold=0.3, circularity_threshold=0.5):
    _, _, w, h, area = stats
    aspect_ratio = min(w, h) / max(w, h)
    
    perimeter = 2 * (w + h)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    
    return abs(1 - aspect_ratio) < aspect_ratio_threshold and circularity > circularity_threshold

# def sum_intensity(stats, bbox):

def devide_objects(centroid1, centroid2):
    # input : centroids of objects
    # output : masks
    slope = (centroid2[1] - centroid1[1]) / (centroid2[0] - centroid1[0])
    perpendicular_slope = -1 / slope
    midpoint = ((centroid1[0] + centroid2[0]) / 2, (centroid1[1] + centroid2[1]) / 2)
    c = midpoint[1] - perpendicular_slope * midpoint[0]
    xx, yy = np.meshgrid(np.arange(0, 25), np.arange(0, 25))
    line_values = perpendicular_slope * xx + c

    mask1 = yy < line_values
    mask2 = yy >= line_values

    return mask1, mask2

if __name__ == '__main__':
    # Load bbox coords
    f = open('/home/yec23006/projects/research/CellDetection/JsonOutput/e2d4.json')
    bboxes = json.load(f)

    # Load image
    image_path = '/home/yec23006/projects/research/CellDetection/MerfishData/DAPI.tif'
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_image_eroded = cv2.erode(gray_image, np.ones((2,2), np.uint8), iterations=1)
    dapi = cv2.dilate(gray_image_eroded, np.ones((4,4), np.uint8), iterations=1)

    cellcnt = 0
    centroid_list = []
    area_list = [] ###########################################

    bbox_plot = cv2.cvtColor(dapi.copy(), cv2.COLOR_GRAY2BGR)
    cell_idx_list = []
    
    for idx in tqdm(range(len(bboxes))):
        bbox = dapi[bboxes[idx]['min_row']:bboxes[idx]['max_row'], bboxes[idx]['min_col']:bboxes[idx]['max_col']]
        bin_bbox = np.where(bbox > np.mean(bbox), 255, 0)

        analysis = cv2.connectedComponentsWithStats(bin_bbox.astype(np.uint8), 4, cv2.CV_32S)
        (totalLabels, label_ids, stats, centroid) = analysis
        ori_centroid = centroid.copy()
        ori_mask = label_ids.copy()
        ori_stats = stats.copy()
        

        if not is_circle(stats[1]):
            i = 3
            loop = 0
            while True:
                kernel = np.ones((i, i), np.uint8)
                erodedImg = cv2.erode(bin_bbox.astype(np.uint8), kernel)
                analysis = cv2.connectedComponentsWithStats(erodedImg, 4, cv2.CV_32S)
                (totalLabels, label_ids, stats, centroid) = analysis
                loop += 1
                
                if totalLabels == 2:
                    i += 1

                elif (totalLabels == 1 and i == 3) or loop >= 10:
                    cellcnt += 1
                    centroid = (ori_centroid[1] + np.array([bboxes[idx]['min_col'], bboxes[idx]['min_row']])).astype(np.int32)
                    centroid_list.append(centroid.astype(np.int32)) 
                    mask1 = np.where(ori_mask==1, 1, 0)
                    area_list.append(ori_stats[1][-1])
                    break

                elif totalLabels == 1 and i >= 4:
                    cellcnt += 1
                    centroid = (centroid[0] + np.array([bboxes[idx]['min_col'], bboxes[idx]['min_row']])).astype(np.int32)  
                    mask1 = np.where(ori_mask==1, 1, 0)
                    centroid_list.append(centroid)
                    area_list.append(ori_stats[1][-1])
                    break

                elif totalLabels == 3:
                    # Two cells overlapped -> get a straight tangent line to get two objects
                    cellcnt += 2
                    centroid1 = (centroid[1] + np.array([bboxes[idx]['min_col'], bboxes[idx]['min_row']])).astype(np.int32)  
                    centroid2 = (centroid[2] + np.array([bboxes[idx]['min_col'], bboxes[idx]['min_row']])).astype(np.int32)  

                    mask1, mask2 = devide_objects(centroid1, centroid2)

                    cv2.circle(bbox_plot, tuple(centroid1), radius=1, color=(0, 0, 255), thickness=-1)
                    cv2.circle(bbox_plot, tuple(centroid2), radius=1, color=(0, 0, 255), thickness=-1)
                    centroid_list.append(centroid1)
                    centroid_list.append(centroid2)
                    area_list.append(stats[1][-1])
                    area_list.append(stats[2][-1])
                    break
                
        else: # if the obj is a circle
            centroid1 = (centroid[1] + np.array([bboxes[idx]['min_col'], bboxes[idx]['min_row']])).astype(np.int32)  
            cv2.circle(bbox_plot, tuple(centroid1), radius=1, color=(0, 0, 255), thickness=-1)
            cellcnt += 1
            centroid_list.append(centroid1)
            area_list.append(stats[1][-1])

    output_path = 'postprocessing/output_image_with_dots_0725.jpeg' 
    cv2.imwrite(output_path, bbox_plot)

    print('Total Cell count : ', cellcnt)

    # Save bounding box coordinates to a JSON file
    centroid_arr = np.array(centroid_list)
    area_arr = np.array(area_list)
    print('Coord, Area saving...')

    with open('postprocessing/coord_0725.npy', 'wb') as f:
        np.save(f, centroid_arr)
    with open('postprocessing/area_0725.npy', 'wb') as f:
        np.save(f, area_arr)

    data = []
    df = pd.DataFrame(data, columns=['Cell id', 'Centroid', 'Area'])
    id_list = ['Cell ' + str(i + 1) for i in range(len(centroid_list))]

    df['Cell id'] = id_list
    df['Centroid'] = centroid_list
    df['DAPI'] = area_list
    df.to_csv('postprocessing/gene_expression.csv', sep='\t')

    print('Gene expression csv saved')

    # Display the output image with dots
    plt.figure(figsize=(10, 10))
    plt.imshow(bbox_plot)
    plt.title(f'Detected Cells: {cellcnt}')
    plt.axis('off')
    plt.show()
