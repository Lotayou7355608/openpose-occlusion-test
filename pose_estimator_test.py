import pose_utils
import os
import numpy as np

from keras.models import load_model
import skimage.transform as st
import pandas as pd
from tqdm import tqdm
from numpy.random import shuffle
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from skimage.io import imsave, imread

from time import time

from imageio import get_reader
from pose_utils import draw_pose_from_cords

from skimage.draw import polygon

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22],
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52],
          [55,56], [37,38], [45,46]]

limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
           [1,16], [16,18], [3,17], [6,18]]

threshold = 0.1
boxsize = 368
scale_search = [0.5, 1, 1.5, 2]


def compute_cordinates(heatmap_avg, paf_avg, oriImg, th1=0.1, th2=0.05):
    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > th1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse

        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = list(range(peak_counter, peak_counter + len(peaks)))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > th2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print("found = 2")
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    if len(subset) == 0:
        return np.array([[-1, -1]] * 18).astype(int)

    cordinates = []
    result_image_index = np.argmax(subset[:, -2])

    for part in subset[result_image_index, :18]:
        if part == -1:
            cordinates.append([-1, -1])
        else:
            Y = candidate[part.astype(int), 0]
            X = candidate[part.astype(int), 1]
            cordinates.append([X, Y])
    return np.array(cordinates).astype(int)


#def cordinates_from_image_file(image_name, model):
#   oriImg = imread(image_name)[:, :, ::-1]  # B,G,R order
def cordinates_from_image_file(image, model):
    oriImg = image[:, :, ::-1]

    multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        new_size = (np.array(oriImg.shape[:2]) * scale).astype(np.int32)
        imageToTest = resize(oriImg, new_size, order=3, preserve_range=True)
        imageToTest_padded = imageToTest[np.newaxis, :, :, :]/255 - 0.5

        output1, output2 = model.predict(imageToTest_padded)

        heatmap = st.resize(output2[0], oriImg.shape[:2], preserve_range=True, order=1)
        paf = st.resize(output1[0], oriImg.shape[:2], preserve_range=True, order=1)
        heatmap_avg += heatmap
        paf_avg += paf

    heatmap_avg /= len(multiplier)
    pose_cords = compute_cordinates(heatmap_avg, paf_avg, oriImg=oriImg)
    return pose_cords

# Estimate 2D-pose for a single image
# output np array coordinates and color images
def estimate_all(folder, model):
    if not os.path.isdir(folder):
        print('Warning: Directory does not exist...')
        return

    pose_folder = folder.replace('images', 'poses')
    if os.path.isdir(pose_folder):
        print('Cached poses found')
    else:
        os.mkdir(pose_folder)
        image_list = [name for name in os.listdir(folder) if name.endswith('.jpg')]
        for name in tqdm(image_list):
            im_name = os.path.join(folder, name)
            img = imread(im_name)
            pose_cords = np.array(cordinates_from_image_file(img, model=model))
            new_path = im_name.replace(folder, pose_folder)
            color, _ = pose_utils.draw_pose_from_cords(pose_cords, (256,256))
            imsave(new_path, color)
            new_path = new_path.replace('.jpg', '.pose.npy')
            np.save(new_path, pose_cords)

    return pose_folder

# Make bad examples by cutting some limbs
def make_bad_images(image_folder, bad_image_folder, limbs, cut_width = 5):
    if not os.path.exists(bad_image_folder):
        os.mkdir(bad_image_folder)

    pose_folder = image_folder.replace('images', 'poses')
    assert(os.path.exists(pose_folder))

    image_list = [name for name in os.listdir(image_folder) if name.endswith('.jpg')]
    for name in tqdm(image_list):
        img = imread(os.path.join(image_folder, name))
        pose = np.load(os.path.join(pose_folder, name.replace('.jpg','.pose.npy')))
        bad_im_name = os.path.join(bad_image_folder, name)
        amputate_limbs(img, pose, bad_im_name, limbs, cut_width)

# Amputate limbs by drawing a black bar at middle point of the bone
# input:
#   @ An image of 256*256
#   @ A corresponding pose locations
#   @ limbs: lthigh, rthigh, lshin, rshin


limbjoints = {'lthigh': (8,9), 'lshin': (9,10), 'rthigh': (11,12), 'rshin': (12,13), 'larm': (2,3), 'rarm': (5,6)}

def amputate_limbs(img, pose, bad_im_name, limbs, bar_width, half_bar_length=20):
    pose = pose.astype(np.float)
    for name in limbs:
        joints = limbjoints[name]
        if any(pose[joints, 0] == -1):
            continue
        
        midpoint = np.sum(pose[joints, :], axis=0) / 2
        direction = np.squeeze(np.array([[1, -1]]) @ pose[joints, :]) # equivalent to p[0] - p[1]
        length = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        direction /= length
        norm_vec = np.array([direction[1], -direction[0]])

        rects = np.stack([midpoint - norm_vec * half_bar_length - direction * bar_width,
                          midpoint - norm_vec * half_bar_length + direction * bar_width,
                          midpoint + norm_vec * half_bar_length + direction * bar_width,
                          midpoint + norm_vec * half_bar_length - direction * bar_width,
                          midpoint - norm_vec * half_bar_length - direction * bar_width]).astype(int)
        rr, cc = polygon(rects[:, 0], rects[:, 1], shape=(256,256))
        rr = np.minimum(np.maximum(rr, 0), 255)
        cc = np.minimum(np.maximum(cc, 0), 255)
        # use opposite color from the middle point
        midpoint = midpoint.astype(int)
        rgb = img[midpoint[0], midpoint[1]]
        img[rr, cc, :] = 255 - rgb

    imsave(bad_im_name, img)


def calculate_missing_rate(pose_folder, bad_pose_folder, dataset_name):
    pose_list = [name for name in os.listdir(pose_folder) if name.endswith('.pose.npy')]
    total_num = len(pose_list)
    rate = 0
    with open('bad_pose_list.txt', 'w') as file:
        for item in pose_list:
            good_pose = np.load(os.path.join(pose_folder, item))
            bad_pose = np.load(os.path.join(bad_pose_folder, item))
            miss_point = any(good_pose[i, 0] >= 0 and bad_pose[i, 0] == -1 for i in range(18))
            if miss_point:
                rate += 1
                file.write('{}\n'.format(item))

    return {dataset_name: rate / total_num}

if __name__ == "__main__":

    # image_folder = 'D:/data/ntu_image_skeleton/all'
    # list = sorted([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
    # number = 200
    # list = list[0:number]
    # dest_folder = 'D:/Yanglingbo_Workspace_new/Projects/Python/TensorFlow/pose-estimator-test/good_ntu_images'
    #
    # from shutil import copyfile
    # for item in list:
    #   copyfile(os.path.join(image_folder, item),
    #            os.path.join(dest_folder, item))

    image_folder = 'good_ntu_images'
    model = load_model('./pose_estimator.h5')
    pose_folder = estimate_all(image_folder, model)
    bad_image_folder = image_folder.replace('good', 'bad')
    make_bad_images(image_folder, bad_image_folder, limbs = ['larm', 'rarm', 'lthigh', 'rthigh'], cut_width = 8)
    bad_pose_folder = estimate_all(bad_image_folder, model)
    rate = calculate_missing_rate(pose_folder, bad_pose_folder, 'ntu')
    for k, v in rate.items():
        print('{} missing rate: {:.3f}'.format(k, v))
