"""
Modul Penghitungan Metrik Evaluasi Deteksi Objek (mAP50, mAP50-95, Precision, Recall).
Mengikuti standar evaluasi COCO/VOC tanpa memerlukan dependensi C-extension pycocotools.
"""

import numpy as np
import torch


def box_iou(boxes1, boxes2):
    """
    Menghitung Intersection over Union (IoU) antara dua set bounding box.
    boxes1: Tensor [N, 4] format [x1, y1, x2, y2]
    boxes2: Tensor [M, 4] format [x1, y1, x2, y2]
    Returns: Tensor [N, M] berisi nilai IoU
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    return iou


def compute_ap(recalls, precisions):
    """
    Menghitung Average Precision (AP) menggunakan kurva Precision-Recall (101-point interpolation).
    """
    # Tambahkan batas awal dan akhir
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Pastikan kurva presisi monotonically decreasing (precision envelope)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation sesuai COCO
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return float(ap)


def evaluate_detections(all_predictions, all_targets, iou_thresholds=None, class_ids=(1, 2)):
    """
    Evaluasi performa deteksi pada seluruh dataset.

    Args:
        all_predictions: list of dict {'boxes': Tensor, 'scores': Tensor, 'labels': Tensor}
        all_targets: list of dict {'boxes': Tensor, 'labels': Tensor}
        iou_thresholds: list nilai IoU (default: [0.50, ..., 0.95])
        class_ids: tuple kelas objek yang dievaluasi (1: cracked, 2: dirt)

    Returns:
        dict: Hasil metrik komprehensif (mAP50, mAP50_95, precision, recall, per_class)
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.50, 0.95, 10).tolist()

    class_names_map = {1: "cracked", 2: "dirt"}
    results_by_iou = {iou_th: {} for iou_th in iou_thresholds}
    precisions_50 = {}
    recalls_50 = {}

    for iou_th in iou_thresholds:
        for cls_id in class_ids:
            cls_name = class_names_map.get(cls_id, f"class_{cls_id}")
            
            # Kumpulkan semua deteksi dan ground truth untuk kelas ini
            detected_scores = []
            detected_matches = []
            num_gt = 0

            for preds, targets in zip(all_predictions, all_targets):
                pred_mask = (preds["labels"] == cls_id)
                gt_mask = (targets["labels"] == cls_id)

                p_boxes = preds["boxes"][pred_mask]
                p_scores = preds["scores"][pred_mask]
                g_boxes = targets["boxes"][gt_mask]

                num_gt += len(g_boxes)

                if len(p_boxes) == 0:
                    continue

                if len(g_boxes) == 0:
                    # Semua deteksi pada gambar tanpa GT adalah False Positive
                    for s in p_scores:
                        detected_scores.append(float(s))
                        detected_matches.append(0)
                    continue

                # Hitung IoU antara deteksi dan ground truth
                ious = box_iou(p_boxes, g_boxes)
                matched_gt = set()

                # Urutkan deteksi berdasarkan confidence score tertinggi
                sort_idx = torch.argsort(p_scores, descending=True)
                for idx in sort_idx:
                    score = float(p_scores[idx])
                    detected_scores.append(score)

                    box_ious = ious[idx]
                    best_iou, best_gt_idx = torch.max(box_ious, dim=0)
                    best_iou = float(best_iou)
                    best_gt_idx = int(best_gt_idx)

                    if best_iou >= iou_th and best_gt_idx not in matched_gt:
                        matched_gt.add(best_gt_idx)
                        detected_matches.append(1)  # True Positive
                    else:
                        detected_matches.append(0)  # False Positive

            if num_gt == 0:
                results_by_iou[iou_th][cls_name] = 0.0
                continue

            if len(detected_scores) == 0:
                results_by_iou[iou_th][cls_name] = 0.0
                if abs(iou_th - 0.50) < 1e-4:
                    precisions_50[cls_name] = 0.0
                    recalls_50[cls_name] = 0.0
                continue

            # Urutkan global berdasarkan skor
            order = np.argsort(-np.array(detected_scores))
            matches = np.array(detected_matches)[order]

            tp_cumsum = np.cumsum(matches)
            fp_cumsum = np.cumsum(1 - matches)

            recalls = tp_cumsum / float(num_gt)
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

            ap = compute_ap(recalls, precisions)
            results_by_iou[iou_th][cls_name] = ap

            if abs(iou_th - 0.50) < 1e-4:
                final_p = float(precisions[-1]) if len(precisions) > 0 else 0.0
                final_r = float(recalls[-1]) if len(recalls) > 0 else 0.0
                precisions_50[cls_name] = final_p
                recalls_50[cls_name] = final_r

    # Hitung rata-rata mAP50 dan mAP50-95
    ap50_list = [results_by_iou[0.50][class_names_map[c]] for c in class_ids]
    map50 = float(np.mean(ap50_list))

    all_aps = []
    for th in iou_thresholds:
        for c in class_ids:
            all_aps.append(results_by_iou[th][class_names_map[c]])
    map50_95 = float(np.mean(all_aps))

    mean_p = float(np.mean(list(precisions_50.values()))) if precisions_50 else 0.0
    mean_r = float(np.mean(list(recalls_50.values()))) if recalls_50 else 0.0

    return {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": mean_p,
        "recall": mean_r,
        "per_class_AP50": {class_names_map[c]: results_by_iou[0.50][class_names_map[c]] for c in class_ids},
        "per_class_AP50_95": {
            class_names_map[c]: float(np.mean([results_by_iou[th][class_names_map[c]] for th in iou_thresholds]))
            for c in class_ids
        }
    }

