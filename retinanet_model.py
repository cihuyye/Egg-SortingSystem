"""
Definisi dan Inisialisasi Model RetinaNet (ResNet-50 FPN) untuk Deteksi Cacat Telur.
Mendukung transfer learning dari bobot COCO pre-trained dan penyesuaian
classification head untuk 2 kelas cacat ('cracked' dan 'dirt') + background.
"""

import os
from functools import partial
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.retinanet import (
    retinanet_resnet50_fpn,
    RetinaNetClassificationHead
)

# Cek ketersediaan RetinaNet v2 dan enum Weights pada Torchvision modern
HAS_RETINANET_V2 = hasattr(torchvision.models.detection, "retinanet_resnet50_fpn_v2")
try:
    from torchvision.models.detection.retinanet import (
        RetinaNet_ResNet50_FPN_Weights,
        RetinaNet_ResNet50_FPN_V2_Weights
    )
    HAS_WEIGHTS_ENUM = True
except ImportError:
    HAS_WEIGHTS_ENUM = False


def create_retinanet_model(
    num_classes=3,
    pretrained=True,
    version="v1",
    score_thresh=0.05,
    nms_thresh=0.50,
    detections_per_img=100
):
    """
    Membangun model RetinaNet dengan backbone ResNet-50-FPN.

    Args:
        num_classes (int): Jumlah kelas total termasuk background (default: 3 -> 0: background, 1: cracked, 2: dirt)
        pretrained (bool): Menggunakan bobot transfer learning dari COCO (default: True)
        version (str): 'v1' (standar klasik) atau 'v2' (arsitektur v2 dengan GroupNorm & pretraining modern)
        score_thresh (float): Batas kepercayaan minimum box deteksi (default: 0.05 saat evaluasi/training)
        nms_thresh (float): IoU threshold untuk Non-Maximum Suppression (default: 0.50)
        detections_per_img (int): Jumlah deteksi maksimal per gambar (default: 100)

    Returns:
        nn.Module: Model RetinaNet Torchvision siap latih / inferensi
    """
    use_v2 = (version.lower() == "v2") and HAS_RETINANET_V2

    if use_v2:
        weights = None
        if pretrained and HAS_WEIGHTS_ENUM:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=weights,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img
        )
        norm_layer = partial(nn.GroupNorm, 32)
    else:
        # Default v1 (kompatibel penuh dengan Torchvision v0.10 - v0.20+)
        weights = None
        if pretrained:
            if HAS_WEIGHTS_ENUM:
                weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
            else:
                # Fallback untuk Torchvision lama
                weights = True

        if HAS_WEIGHTS_ENUM:
            model = retinanet_resnet50_fpn(
                weights=weights,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                detections_per_img=detections_per_img
            )
        else:
            model = retinanet_resnet50_fpn(
                pretrained=(pretrained is True),
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                detections_per_img=detections_per_img
            )
        norm_layer = None

    # Modifikasi Classification Head agar sesuai dengan jumlah kelas kustom kita (3 kelas)
    in_channels = getattr(model.head.classification_head, "in_channels", getattr(model.head.classification_head.cls_logits, "in_channels", 256))
    num_anchors = getattr(model.head.classification_head, "num_anchors", 9)

    if norm_layer is not None:
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=norm_layer,
            prior_probability=0.01
        )
    else:
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            prior_probability=0.01
        )

    return model


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Menyimpan checkpoint model dan metadata training"""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "class_names": ["__background__", "cracked", "dirt"],
        "num_classes": 3,
        "architecture": "retinanet_resnet50_fpn"
    }
    torch.save(checkpoint, save_path)
    print(f"[*] Checkpoint berhasil disimpan ke: {save_path}")


def load_checkpoint(weights_path, device="cpu", score_thresh=0.40):
    """
    Memuat model RetinaNet dari file checkpoint .pth
    
    Args:
        weights_path (str): Path ke file .pth
        device (str): Perangkat target ('cpu' atau 'cuda')
        score_thresh (float): Threshold skor prediksi untuk inferensi
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"File bobot tidak ditemukan di {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 3)

    # Buat instance model tanpa download bobot COCO lagi
    model = create_retinanet_model(
        num_classes=num_classes,
        pretrained=False,
        score_thresh=score_thresh
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    class_names = checkpoint.get("class_names", ["__background__", "cracked", "dirt"])
    return model, class_names

