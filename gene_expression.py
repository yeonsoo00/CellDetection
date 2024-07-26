import json
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd 
import os

if __name__ == "__main__":

    # load image and data
    gene_list = ['CHI3L1', 'CHI3L2', 'CILP', 'COL1A2', 'COL2A1', 'COL3A1', 'CRTAC1', 'PRG4', 'STC2']
    path2dir = '/home/yec23006/projects/research/CellDetection/MerfishData/'
    df = pd.read_csv('postprocessing/gene_expression.csv', sep = '\t', index_col=False)
    centroid_list = df['Centroid'].to_list()
    bbox_list = df['Bbox'].to_list()

    for gene in tqdm(gene_list):
        img = cv2.imread(os.path.join(path2dir, gene + '.tif'), cv2.IMREAD_GRAYSCALE)
        img = cv2.erode(img, np.ones((2,2), np.uint8), iterations=1)
        img = cv2.dilate(img, np.ones((4,4), np.uint8), iterations=1)

        gene_expression = []

        for idx, bbox in enumerate(bbox_list):
            temp = bbox.strip('][').split(', ')
            bbox = [int(x) for x in temp]

            centroid = centroid_list[idx]
            image_patch = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            gene_expression.append(np.sum(image_patch))

        df[gene] = gene_expression

df.to_csv('CellDetection/Results/gene_expression.csv', sep='\t', index=False)
print('Gene expression csv saved')