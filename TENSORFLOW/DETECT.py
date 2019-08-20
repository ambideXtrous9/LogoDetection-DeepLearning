import sys
import tensorflow as tf
import numpy as np
from scipy.misc import imresize
import selectivesearch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import MODEL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PIXEL_DEPTH = 255.0
PRED_PROB_THRESH = 0.999

def get_object_proposals(img,scale=500, sigma=0.9, min_size=10):
    # Selective search
    img_lbl, regions = selectivesearch.selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 500 pixels
        x, y, w, h = r['rect']
        if r['size'] < 500 or w > 0.95 * img.shape[1] or h > 0.95 * img.shape[0]:
            continue
        # excluding the zero-width or zero-height box
        if r['rect'][2] == 0 or r['rect'][3] == 0:
            continue
        # distorted rects
        if w / h > 5 or h / w > 5:
            continue
        candidates.add(r['rect'])

    return candidates

def load_target_image(img_fn):
    target_image = cv2.imread(img_fn)
    image = cv2.cvtColor(target_image,cv2.COLOR_BGR2RGB)
    return image


def update_idx(results):
    probs = np.array([r['pred_prob'] for r in results])
    idx = np.argsort(probs)[::-1]
    return idx


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_fn', help='image filename')
    return parser.parse_args()

def scaling(image_data):
    return (image_data.astype(np.float32) - (PIXEL_DEPTH / 2)) / PIXEL_DEPTH

def iou_xywh(box1, box2):

    #box1 -- rectangles of object proposals with coordinates (x, y, w, h)
    #box2 -- rectangle of ground truth with coordinates (x1, y1, w, h)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[0] + box1[2], box2[0] + box2[2])
    yi2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)

    # Calculate the union area by using formula: union(A, B) = A + B - inter_area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou



def nms(recog_results, pred_prob_th=0.99, iou_th=0.5):
    # nms results
    nms_results = []

    # Discard all results with prob <= pred_prob_th
    pred_probs = np.array([r['pred_prob'] for r in recog_results])
    cand_idx = np.where(pred_probs > pred_prob_th)[0]
    cand_results = np.array(recog_results)[cand_idx]
    if len(cand_results) == 0:
        return nms_results

    # Sort in descending order
    cand_nms_idx = update_idx(cand_results)
    # Pick the result with the largest prob as a prediction
    pred = cand_results[cand_nms_idx[0]]
    nms_results.append(pred)
    if len(cand_results) == 1:
        return nms_results
    cand_results = cand_results[cand_nms_idx[1:]]
    cand_nms_idx = update_idx(cand_results)

    # Discard any remaining results with IoU >= iou_th
    while len(cand_results) > 0:
        del_idx = []
        del_seq_idx = []
        for seq_i, i in enumerate(cand_nms_idx):
            if iou_xywh(cand_results[i]['obj_proposal'],pred['obj_proposal']) >= iou_th:
                del_idx.append(i)
                del_seq_idx.append(seq_i)
        # Delete non-max results
        cand_results = np.delete(cand_results, del_idx)
        if len(cand_results) == 0:
            break
        cand_nms_idx = update_idx(cand_results)
        # For next iteration
        pred, cand_results = cand_results[cand_nms_idx[0]], cand_results[
            cand_nms_idx[1:]]
        if len(cand_results) == 0:
            break
        cand_nms_idx = update_idx(cand_results)
        nms_results.append(pred)

    return nms_results

def logo_recognition(sess, img, obj_proposal, graph_params):
    # recognition results
    recog_results = {}
    recog_results['obj_proposal'] = obj_proposal
    # Resize image
    if img.shape != MODEL.CNN_SHAPE:
        img = imresize(img, MODEL.CNN_SHAPE, interp='bicubic')
    # Pre-processing
    img = scaling(img)
    img = img.reshape((1, MODEL.CNN_IN_HEIGHT, MODEL.CNN_IN_WIDTH,MODEL.CNN_IN_CH)).astype(np.float32)
    # Logo recognition
    pred = sess.run([graph_params['pred']], feed_dict={graph_params['target_image']: img})
    recog_results['pred_class'] = MODEL.CLASS_NAME[np.argmax(pred)]
    recog_results['pred_prob'] = np.max(pred)
    return recog_results


def setup_graph():
    graph_params = {}
    graph_params['graph'] = tf.Graph()
    
    with graph_params['graph'].as_default():
        
        model_params = MODEL.params()
        graph_params['target_image'] = tf.placeholder(tf.float32,shape=(1, MODEL.CNN_IN_HEIGHT, MODEL.CNN_IN_WIDTH,MODEL.CNN_IN_CH))
        logits = MODEL.CNN(graph_params['target_image'], model_params, keep_prob=1.0)
        graph_params['pred'] = tf.nn.softmax(logits)
        graph_params['saver'] = tf.train.Saver()
    
    return graph_params

def main():
    args = parse_cmdline()
    img_fn = os.path.abspath(args.img_fn)
    if not os.path.exists(img_fn):
        print('Not found: {}'.format(img_fn))
        sys.exit(-1)
    else:
        print('Target image: {}'.format(img_fn))

    # Load target image
    target_image = load_target_image(img_fn)

    # Get object proposals
    object_proposals = get_object_proposals(target_image)

    # Setup computation graph
    graph_params = setup_graph()

    # Model initialize
    sess = tf.Session(graph=graph_params['graph'])

    tf.global_variables_initializer()
    if os.path.exists('train_models'):
        save_path = os.path.join('train_models', 'deep_logo_model')
        graph_params['saver'].restore(sess, save_path)
        print('Model restored')
    else:
        print('Initialized')

    # Logo recognition
    results = []
    for obj_proposal in object_proposals:
        x, y, w, h = obj_proposal
        crop_image = target_image[y:y + h, x:x + w]
        results.append(logo_recognition(sess, crop_image, obj_proposal, graph_params))

    del_idx = []
    for i, result in enumerate(results):
        if result['pred_class'] == MODEL.CLASS_NAME[-1]:
            del_idx.append(i)
    results = np.delete(results, del_idx)

    nms_results = nms(results, PRED_PROB_THRESH, iou_th=0.7)
    target_image = cv2.cvtColor(target_image,cv2.COLOR_BGR2RGB)

    for result in nms_results:
        print(result)
        (x, y, w, h) = result['obj_proposal']
        cv2.putText(target_image,result['pred_class'],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("RESULT",target_image)
    cv2.waitKey()



if __name__ == '__main__':
    main()
