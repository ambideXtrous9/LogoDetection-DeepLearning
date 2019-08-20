import numpy as np
import selectivesearch
import skimage.io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

__all__ = ('CLASS_NAME', 'CNN_IN_WIDTH', 'CNN_IN_HEIGHT', 'CNN_IN_CH',
           'CNN_SHAPE', 'TRAIN_DIR', 'TRAIN_IMAGE_DIR',
           'CROPPED_AUG_IMAGE_DIR', 'ANNOT_FILE', 'ANNOT_FILE_WITH_BG')


CLASS_NAME = [
    'Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
    'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds', 'Mini', 'Nbc',
    'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks',
    'Texaco', 'Unicef', 'Vodafone', 'Yahoo', 'Background']

CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32
CNN_IN_CH = 3
CNN_SHAPE = (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)

TRAIN_DIR = 'flickr_logos_27_dataset'
TRAIN_IMAGE_DIR = os.path.join('flickr_logos_27_dataset_images')
CROPPED_AUG_IMAGE_DIR = os.path.join('flickr_logos_27_dataset_cropped_augmented_images')
ANNOT_FILE = os.path.join('flickr_logos_27_dataset_training_set_annotation.txt')
ANNOT_FILE_WITH_BG = os.path.join('train_annot_with_bg_class.txt')

def parse_annot(annot):
    fn = annot[0].decode('utf-8')
    class_name = annot[1].decode('utf-8')
    train_subset_class = annot[2].decode('utf-8')
    return fn, class_name, train_subset_class

def get_annot_rect(annot):
    return np.array(list(map(lambda x: int(x), annot[3:])))


def iou(obj_proposal, annot_rect):
    #obj_proposals -- rectangles of object proposals with coordinates (x, y, w, h)
    #annot_rect -- rectangle of ground truth with coordinates (x1, y1, x2, y2)

    xi1 = max(obj_proposal[0], annot_rect[0])
    yi1 = max(obj_proposal[1], annot_rect[1])
    xi2 = min(obj_proposal[0] + obj_proposal[2], annot_rect[2])
    yi2 = min(obj_proposal[1] + obj_proposal[3], annot_rect[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)

    # Calculate the union area by using formula: union(A, B) = A + B - inter_area
    box1_area = obj_proposal[2] * obj_proposal[3]
    box2_area = (annot_rect[2] - annot_rect[0]) * (
        annot_rect[3] - annot_rect[1])
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou

def get_bg_proposals(object_proposals, annot):
    annot_rect = get_annot_rect(annot)
    bg_proposals = []
    for obj_proposal in object_proposals:
        if iou(obj_proposal, annot_rect) <= 0.5:
            bg_proposals.append(obj_proposal)
    return bg_proposals
    
def get_object_proposals(img, scale=500, sigma=0.9, min_size=10):
    # Selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=scale, sigma=sigma, min_size=min_size)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 500 pixels
        x, y, w, h = r['rect']
        if r['size'] < 2000 or w > 0.95 * img.shape[1] or h > 0.95 * img.shape[0]:
            continue
        # excluding the zero-width or zero-height box
        if r['rect'][2] == 0 or r['rect'][3] == 0:
            continue
        # distorted rects
        if w / h > 5 or h / w > 5:
            continue
        candidates.add(r['rect'])

    return candidates



def gen_annot_file_line(img_fn, class_name, train_subset_class, rect):
    rect = ' '.join(map(str, rect))
    line = ' '.join([img_fn, class_name, train_subset_class, rect])
    return line


def gen_annot_file_lines(annot):
    lines = []

    # Get original annot line
    img_fn, class_name, train_subset_class = parse_annot(annot)
    annot_rect = get_annot_rect(annot)
    lines.append( gen_annot_file_line(img_fn, class_name, train_subset_class, annot_rect))

    # Load image
    img = skimage.io.imread(os.path.join(TRAIN_IMAGE_DIR, img_fn))

    # Selective search
    object_proposals = get_object_proposals(img)
    if len(object_proposals) == 0:
        return lines

    # Background proposals
    bg_proposals = get_bg_proposals(object_proposals, annot)
    if len(bg_proposals) == 0:
        return lines

    # Select bg proposal
    bg_proposal = bg_proposals[np.random.choice( np.array(bg_proposals).shape[0])]

    x1, y1, x2, y2 = bg_proposal[0], bg_proposal[1], bg_proposal[0] + bg_proposal[2], bg_proposal[1] + bg_proposal[3]
    lines.append(gen_annot_file_line(img_fn, CLASS_NAME[-1], train_subset_class,[x1, y1, x2, y2]))

    return lines


def main():
    # Load an annotation file
    annot_train = np.loadtxt(ANNOT_FILE, dtype='a')
    print('train_annotation: {}'.format(annot_train.shape[0]))

    # Multi processing
    results = []
    n_workers = os.cpu_count()

    with ProcessPoolExecutor(n_workers) as executer, open(ANNOT_FILE_WITH_BG, 'w') as fw:
        for annot in annot_train:
            results.append(executer.submit(gen_annot_file_lines, annot))

        for result in as_completed(results):
            print('\n'.join(result.result()))
            fw.writelines('\n'.join(result.result()))
            fw.writelines('\n')


if __name__ == '__main__':
    main()
