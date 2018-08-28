import numpy as np
import tensorflow as tf

from workshop.resnet import resnet_v1_101, resnet_v1_101_tail


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


OUTPUT_STRIDE = 16

CLASS_NMS_THRESHOLD = 0.5
TOTAL_MAX_DETECTIONS = 300


def sort_anchors(anchors):
    """Sort the anchor references aspect ratio first, then area."""
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    aspect_ratios = np.round(heights / widths, 1)
    areas = widths * heights

    return anchors[np.lexsort((areas, aspect_ratios)), :]


def change_order(bboxes):
    first_min, second_min, first_max, second_max = tf.unstack(
        bboxes, axis=1
    )
    bboxes = tf.stack(
        [second_min, first_min, second_max, first_max], axis=1
    )
    return bboxes


def get_width_upright(bboxes):
    bboxes = tf.cast(bboxes, tf.float32)
    x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
    width = x2 - x1
    height = y2 - y1

    # Calculate up right point of bbox (urx = up right x)
    urx = x1 + .5 * width
    ury = y1 + .5 * height

    return width, height, urx, ury


def clip_boxes(bboxes, imshape):
    """
    Clips bounding boxes to image boundaries based on image shape.

    Args:
        bboxes: Tensor with shape (num_bboxes, 4)
            where point order is x1, y1, x2, y2.

        imshape: Tensor with shape (2, )
            where the first value is height and the next is width.

    Returns
        Tensor with same shape as bboxes but making sure that none
        of the bboxes are outside the image.
    """
    bboxes = tf.cast(bboxes, dtype=tf.float32)
    imshape = tf.cast(imshape, dtype=tf.float32)

    x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
    width = imshape[1]
    height = imshape[0]
    x1 = tf.maximum(tf.minimum(x1, width - 1.0), 0.0)
    x2 = tf.maximum(tf.minimum(x2, width - 1.0), 0.0)

    y1 = tf.maximum(tf.minimum(y1, height - 1.0), 0.0)
    y2 = tf.maximum(tf.minimum(y2, height - 1.0), 0.0)

    bboxes = tf.concat([x1, y1, x2, y2], axis=1)

    return bboxes


def run_base_network(inputs):
    """Obtain the feature map for an input image."""
    # Pre-process inputs as required by the Resnet (just substracting means).
    means = tf.constant([_R_MEAN, _G_MEAN, _B_MEAN], dtype=tf.float32)
    processed_inputs = inputs - means

    _, endpoints = resnet_v1_101(
        processed_inputs,
        training=False,
        global_pool=False,
        output_stride=OUTPUT_STRIDE,
    )

    feature_map = endpoints['resnet_v1_101/block3']

    return feature_map


def run_resnet_tail(inputs):
    """Pass `inputs` through the last block of the Resnet.

    Arguments:
        inputs: Tensor of shape (total_proposals, pool_size, pool_size, 1024),
            the result of the RoI pooling layer.

    Returns:
        Tensor of shape (total_proposals, pool_size, pool_size, 2048), with the
        output of the final block.
    """
    return resnet_v1_101_tail(inputs)[0]


def decode(roi, deltas):
    (
        roi_width, roi_height, roi_urx, roi_ury
    ) = get_width_upright(roi)

    dx, dy, dw, dh = tf.split(deltas, 4, axis=1)

    pred_ur_x = dx * roi_width + roi_urx
    pred_ur_y = dy * roi_height + roi_ury
    pred_w = tf.exp(dw) * roi_width
    pred_h = tf.exp(dh) * roi_height

    bbox_x1 = pred_ur_x - 0.5 * pred_w
    bbox_y1 = pred_ur_y - 0.5 * pred_h

    bbox_x2 = pred_ur_x + 0.5 * pred_w
    bbox_y2 = pred_ur_y + 0.5 * pred_h

    bboxes = tf.concat([
        bbox_x1, bbox_y1, bbox_x2, bbox_y2
    ], axis=1)

    return bboxes


def rcnn_proposals(proposals, bbox_pred, cls_prob, im_shape, num_classes,
                   min_prob_threshold=0.0, class_max_detections=100):
    """
    Args:
        proposals: Tensor with the RPN proposals bounding boxes.
            Shape (num_proposals, 4). Where num_proposals is less than
            POST_NMS_TOP_N (We don't know exactly beforehand)
        bbox_pred: Tensor with the RCNN delta predictions for each proposal
            for each class. Shape (num_proposals, 4 * num_classes)
        cls_prob: A softmax probability for each proposal where the idx = 0
            is the background class (which we should ignore).
            Shape (num_proposals, num_classes + 1)

    Returns:
        objects:
            Shape (final_num_proposals, 4)
            Where final_num_proposals is unknown before-hand (it depends on
            NMS). The 4-length Tensor for each corresponds to:
            (x_min, y_min, x_max, y_max).
        objects_label:
            Shape (final_num_proposals,)
        objects_label_prob:
            Shape (final_num_proposals,)

    """
    selected_boxes = []
    selected_probs = []
    selected_labels = []

    TARGET_VARIANCES = np.array([0.1, 0.1, 0.2, 0.2])

    # For each class, take the proposals with the class-specific
    # predictions (class scores and bbox regression) and filter accordingly
    # (valid area, min probability score and NMS).
    for class_id in range(num_classes):
        # Apply the class-specific transformations to the proposals to
        # obtain the current class' prediction.
        class_prob = cls_prob[:, class_id + 1]  # 0 is background class.
        class_bboxes = bbox_pred[:, (4 * class_id):(4 * class_id + 4)]
        raw_class_objects = decode(
            proposals,
            class_bboxes * TARGET_VARIANCES,
        )

        # Clip bboxes so they don't go out of the image.
        class_objects = clip_boxes(raw_class_objects, im_shape)

        # Filter objects based on the min probability threshold and on them
        # having a valid area.
        prob_filter = tf.greater_equal(class_prob, min_prob_threshold)

        (x_min, y_min, x_max, y_max) = tf.unstack(class_objects, axis=1)
        area_filter = tf.greater(
            tf.maximum(x_max - x_min, 0.0)
            * tf.maximum(y_max - y_min, 0.0),
            0.0
        )

        object_filter = tf.logical_and(area_filter, prob_filter)

        class_objects = tf.boolean_mask(class_objects, object_filter)
        class_prob = tf.boolean_mask(class_prob, object_filter)

        # We have to use the TensorFlow's bounding box convention to use
        # the included function for NMS.
        class_objects_tf = change_order(class_objects)

        # Apply class NMS.
        class_selected_idx = tf.image.non_max_suppression(
            class_objects_tf, class_prob, class_max_detections,
            iou_threshold=CLASS_NMS_THRESHOLD
        )

        # Using NMS resulting indices, gather values from Tensors.
        class_objects_tf = tf.gather(class_objects_tf, class_selected_idx)
        class_prob = tf.gather(class_prob, class_selected_idx)

        # Revert to our bbox convention.
        class_objects = change_order(class_objects_tf)

        # We append values to a regular list which will later be
        # transformed to a proper Tensor.
        selected_boxes.append(class_objects)
        selected_probs.append(class_prob)
        # In the case of the class_id, since it is a loop on classes, we
        # already have a fixed class_id. We use `tf.tile` to create that
        # Tensor with the total number of indices returned by the NMS.
        selected_labels.append(
            tf.tile([class_id], [tf.shape(class_selected_idx)[0]])
        )

    # We use concat (axis=0) to generate a Tensor where the rows are
    # stacked on top of each other
    objects = tf.concat(selected_boxes, axis=0)
    proposal_label = tf.concat(selected_labels, axis=0)
    proposal_label_prob = tf.concat(selected_probs, axis=0)

    # Get top-k detections of all classes.
    k = tf.minimum(
        TOTAL_MAX_DETECTIONS,
        tf.shape(proposal_label_prob)[0]
    )

    top_k = tf.nn.top_k(proposal_label_prob, k=k)
    top_k_proposal_label_prob = top_k.values
    top_k_objects = tf.gather(objects, top_k.indices)
    top_k_proposal_label = tf.gather(proposal_label, top_k.indices)

    return (
        top_k_objects,
        top_k_proposal_label,
        top_k_proposal_label_prob,
    )
