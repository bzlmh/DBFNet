import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
import csv

def binary_pa(s, g):
    """
    Calculate the pixel accuracy of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    pa = ((s == g).sum()) / g.size
    return pa


def cal_ssim(im1, im2):
    assert im1.shape == im2.shape
    if len(im1.shape) > 2:
        im1 = np.mean(im1, axis=2)
        im2 = np.mean(im2, axis=2)
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim_val = l12 * c12 * s12
    return ssim_val


def tp(original, contrast):
    return np.sum(np.logical_and(original == 0, contrast == 0))


def fp(original, contrast):
    return np.sum(np.logical_and(original == 0, contrast != 0))


def fn(original, contrast):
    return np.sum(np.logical_and(original != 0, contrast == 0))


def recall(TP, FN):
    if TP + FN == 0:
        return 0
    recall = ((TP) / (TP + FN)) * 100
    return recall


def precision(TP, FP):
    if TP + FP == 0:
        return 0
    precision = ((TP) / (TP + FP)) * 100
    return precision


def f1_measure(precision, recall):
    if precision + recall == 0:
        return 0
    f1_measure = 2 * precision * recall / (precision + recall)
    return f1_measure


def cal_pseudo_f_measure(im1, im2):
    TP = tp(im1, im2)
    FP = fp(im1, im2)
    FN = fn(im1, im2)

    pseudo_f_measure = (2 * TP) / (2 * TP + FP + FN)

    return pseudo_f_measure


def process_images(gt_path, binarized_path, threshold=128):
    results = []
    for gt_file in os.listdir(gt_path):
        if gt_file.endswith(".png"):
            gt_img = np.array(Image.open(os.path.join(gt_path, gt_file)).convert('L'))
            binarized_img = np.array(Image.open(os.path.join(binarized_path, gt_file)).convert('L'))

            # 设定阈值
            binarized_img[binarized_img <= threshold] = 0
            binarized_img[binarized_img > threshold] = 255

            # Calculate PSNR
            psnr_val = psnr(gt_img, binarized_img)

            # Calculate SSIM
            ssim_val = cal_ssim(gt_img, binarized_img)

            # Calculate Pixel Accuracy
            pa = binary_pa(binarized_img, gt_img)

            # Calculate F-Measure
            TP = tp(gt_img, binarized_img)
            FP = fp(gt_img, binarized_img)
            FN = fn(gt_img, binarized_img)
            precision_val = precision(TP, FP)
            recall_val = recall(TP, FN)
            f_measure = f1_measure(precision_val, recall_val)

            # Calculate Pseudo F-Measure
            pseudo_f_measure = cal_pseudo_f_measure(gt_img, binarized_img)

            # Append to results
            results.append([gt_file, "%.2f" % f_measure, "%.2f" % pseudo_f_measure, "%.2f" % psnr_val, "%.2f" % ssim_val, "%.2f" % pa])

    return results


if __name__ == "__main__":
    gt_path = 'results/merge_gt'
    binarized_path = 'refine_out'

    results = process_images(gt_path, binarized_path)

    # Save results to CSV file
    csv_file = "segmentation_results.csv"
    header = ["Image Name", "F-Measure", "Pseudo F-Measure", "PSNR", "SSIM", "Pixel Accuracy"]

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)

    print("Results saved to:", csv_file)
