import numpy as np


def compute_iou(box:np.ndarray, boxes:np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0],box[:,0])
    inter_y1 = np.maximum(box[1],box[:,1])
    inter_x2 = np.maximum(box[2],box[:,2])
    inter_y2 = np.maximum(box[3],box[:,3])

    inter_w = np.maximum(0,inter_x2 - inter_x1)
    inter_h = np.maximum(0,inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = boxes[:,2] - boxes[:,0] * boxes[:,3] - boxes[:,1]

    union_area = box_area + boxes_area - inter_area

    ious = inter_area / np.maximum(union_area,1e-6)
    return ious

def nms(boxes:np.ndarray,scores:np.ndarray,ious_threshold:float=0.5) -> np.ndarray:
    if len(boxes)==0:
        return np.array([],dtype=np.int32)
    sorted_indices = np.argsort(scores)[::-1]
    keep = []

    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        if len(sorted_indices) == 1:
            break
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]
        ious = compute_iou(boxes[current_idx],remaining_boxes)
        sorted_indices = remaining_indices[ious < ious_threshold]
    return np.array(keep,dtype=np.int32)