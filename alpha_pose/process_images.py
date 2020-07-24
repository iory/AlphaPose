from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import gdown

from alpha_pose import InferenNet_fast
from alpha_pose import dataloader
from alpha_pose import Darknet
from alpha_pose import yolo_utils


def get_yolo_model():
    return gdown.cached_download(
        'https://drive.google.com/uc?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC',
        md5='6c569c9ef748131c2d96b2deab446a5f',
        quiet=True)


def get_sppe_model():
    return gdown.cached_download(
        'https://drive.google.com/uc?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW',
        md5='414d2b4e20ebfbbdac41780b3730262e',
        quiet=True)


class AlphaPose(nn.Module):

    def __init__(self):
        super(AlphaPose, self).__init__()

        pose_dataset = dataloader.Mscoco()
        self.pose_model = InferenNet_fast(
            4 * 1 + 1, pose_dataset, get_sppe_model())
        self.pose_model.eval()

        self.det_model = Darknet()
        self.det_model.load_weights(get_yolo_model())
        self.det_model.eval()

        self.det_model.net_info['height'] = 608
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert(self.det_inp_dim % 32 == 0)
        assert(self.det_inp_dim > 32)

    def _get_device_type(self):
        return next(self.parameters()).device

    def process(self, bgr_imgs, nms_thesh=0.6, confidence=0.05):
        """Process human pose estimation.

        Parameters
        ----------
        bgr_imgs : list[numpy.ndarray]
            list of input images.
        """

        device = self._get_device_type()
        num_classes = 80
        results_list = []
        for frame in bgr_imgs:
            img, orig_img, inp, im_dim_list = \
                dataloader.one_frame_prepare(frame)
            with torch.no_grad():
                # Human Detection
                img = img.to(device)
                im_dim_list = im_dim_list.to(device)

                prediction = self.det_model(img)
                # NMS process
                dets = yolo_utils.dynamic_write_results(
                    prediction, confidence,
                    num_classes, nms=True, nms_conf=nms_thesh)
                results = {'result': []}
                if isinstance(dets, int) or dets.shape[0] == 0:
                    pass
                else:
                    im_dim_list = torch.index_select(
                        im_dim_list, 0, dets[:, 0].long())
                    scaling_factor = torch.min(
                        self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                    # coordinate transfer
                    dets[:, [1, 3]] -= (
                        self.det_inp_dim -
                        scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                    dets[:, [2, 4]] -= (
                        self.det_inp_dim -
                        scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                    dets[:, 1:5] /= scaling_factor
                    for j in range(dets.shape[0]):
                        dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0,
                                                      im_dim_list[j, 0])
                        dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0,
                                                      im_dim_list[j, 1])
                    boxes = dets[:, 1:5].cpu()
                    scores = dets[:, 5:6].cpu()
                    # Pose Estimation
                    inputResH = 320
                    inputResW = 256
                    inps = torch.zeros(boxes.size(0), 3, inputResH, inputResW)
                    pt1 = torch.zeros(boxes.size(0), 2)
                    pt2 = torch.zeros(boxes.size(0), 2)
                    inps, pt1, pt2 = dataloader.crop_from_dets(
                        inp, boxes, inps, pt1, pt2)
                    inps = inps.to(device)
                    hm = self.pose_model(inps)
                    results = dataloader.get_result(
                        boxes, scores, hm.cpu(),
                        pt1, pt2, orig_img)
                    for j, human in enumerate(results['result']):
                        kp_preds = human['keypoints']
                        kp_scores = human['kp_score']
                        score = human['proposal_score']
                        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
                        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))
                        human['keypoints'] = kp_preds.cpu().numpy()
                        human['kp_score'] = kp_scores.cpu().numpy()
                        human['proposal_score'] = float(score)
            results_list.append(results['result'])
        return results_list


def process_images(bgr_imgs,
                   nms_thesh=0.6,
                   confidence=0.05):
    pose_dataset = dataloader.Mscoco()
    pose_model = InferenNet_fast(
        4 * 1 + 1, pose_dataset, get_sppe_model())
    det_model = Darknet()
    det_model.load_weights(get_yolo_model())

    det_model.net_info['height'] = 608
    det_inp_dim = int(det_model.net_info['height'])
    assert(det_inp_dim % 32 == 0)
    assert(det_inp_dim > 32)
    det_model.cuda()
    det_model.eval()

    pose_model.cuda()
    pose_model.eval()

    num_classes = 80

    results_list = []
    for frame in bgr_imgs:
        img, orig_img, inp, im_dim_list = \
            dataloader.one_frame_prepare(frame)
        with torch.no_grad():
            # Human Detection
            img = Variable(img).cuda()
            im_dim_list = im_dim_list.cuda()

            prediction = det_model(img, CUDA=True)
            # NMS process
            dets = yolo_utils.dynamic_write_results(
                prediction, confidence,
                num_classes, nms=True, nms_conf=nms_thesh)
            results = {'result': []}
            if isinstance(dets, int) or dets.shape[0] == 0:
                pass
            else:
                im_dim_list = torch.index_select(
                    im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(
                    det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (
                    det_inp_dim -
                    scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (
                    det_inp_dim -
                    scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0,
                                                  im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0,
                                                  im_dim_list[j, 1])
                boxes = dets[:, 1:5].cpu()
                scores = dets[:, 5:6].cpu()
                # Pose Estimation
                inputResH = 320
                inputResW = 256
                inps = torch.zeros(boxes.size(0), 3, inputResH, inputResW)
                pt1 = torch.zeros(boxes.size(0), 2)
                pt2 = torch.zeros(boxes.size(0), 2)
                inps, pt1, pt2 = dataloader.crop_from_dets(
                    inp, boxes, inps, pt1, pt2)
                inps = Variable(inps.cuda())
                hm = pose_model(inps)
                results = dataloader.get_result(
                    boxes, scores, hm.cpu(),
                    pt1, pt2, orig_img)
        results_list.append(results)
    return results_list


def add_3d_positions(results,
                     depth_imgs,
                     cameramodel):
    for result, depth_img in zip(results, depth_imgs):
        if len(result['result']) == 0:
            continue
        for human in result['result']:
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            kp_preds = torch.cat(
                (kp_preds,
                 torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat(
                (kp_scores,
                 torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            human['keypoints_3d'] = np.zeros((kp_scores.shape[0], 4), 'f')
            for n in range(kp_scores.shape[0]):
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                if not (0 <= cor_x < cameramodel.width and
                        0 <= cor_y < cameramodel.height):
                    human['keypoints_3d'][n][3] = 1.0
                    continue
                xyz = cameramodel.project_pixel_to_3d_ray(
                    np.array([cor_x, cor_y]))
                z = depth_img[cor_y, cor_x]
                x = xyz[0] * z
                y = xyz[1] * z
                human['keypoints_3d'][n][:3] = np.array([x, y, z], 'f')


def extract_3d_positions(results, index,
                         score_thresh=0.05):
    positions = []
    for result in results:
        cur_pos = []
        for human in result['result']:
            kp_scores = human['kp_score']
            kp_preds = human['keypoints_3d']
            for n in range(kp_scores.shape[0]):
                if n != index:
                    continue
                if kp_scores[n] <= score_thresh:
                    continue
                # ignore out-frame point.
                if kp_preds[n][3] == 1.0:
                    continue
                cur_pos.append(kp_preds[n][:3])
        positions.append(cur_pos)
    return positions
