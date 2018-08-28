import tensorflow as tf

from collections import namedtuple


class Block(namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
          returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The
          list contains one (depth, depth_bottleneck, stride) tuple for each
          unit in the block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.

    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.

    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with
          the input, either intact (if factor == 1) or subsampled (if factor >
          1).
    """
    if factor == 1:
        return inputs

    with tf.variable_scope(scope):
        return tf.layers.max_pooling2d(
            inputs, [1, 1], strides=factor, padding='same'
        )


def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN after convolutions.

    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2]
    for its definition. Note that we use here the bottleneck variant which has
    an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling
        of the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as scope:
        depth_in = inputs.get_shape()[-1].value
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            with tf.variable_scope('shortcut'):
                pre_shortcut = tf.layers.conv2d(
                    inputs, depth, [1, 1], strides=stride, use_bias=False,
                    padding='same',
                )
                shortcut = tf.layers.batch_normalization(
                    pre_shortcut, momentum=0.997, epsilon=1e-5, training=False,
                    fused=False
                )

        with tf.variable_scope('conv1'):
            residual = tf.layers.conv2d(
                inputs, depth_bottleneck, [1, 1], strides=1, use_bias=False,
                padding='same',
            )
            residual = tf.layers.batch_normalization(
                residual, momentum=0.997, epsilon=1e-5, training=False,
                fused=False
            )
            residual = tf.nn.relu(residual)

        with tf.variable_scope('conv2'):
            residual = conv2d_same(
                residual, depth_bottleneck, 3, strides=stride,
                dilation_rate=rate,
            )
            residual = tf.layers.batch_normalization(
                residual, momentum=0.997, epsilon=1e-5, training=False,
                fused=False
            )
            residual = tf.nn.relu(residual)

        with tf.variable_scope('conv3'):
            residual = tf.layers.conv2d(
                residual, depth, [1, 1], strides=1, use_bias=False,
                padding='same',
            )
            residual = tf.layers.batch_normalization(
                residual, momentum=0.997, epsilon=1e-5, training=False,
                fused=False
            )

        output = tf.nn.relu(shortcut + residual)

        return output


def resnet_v1_block(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v1 bottleneck block.

    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last
        unit. All other units have stride=1.

    Returns:
      A resnet_v1 bottleneck block.
    """
    return Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
     }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
      }])


def conv2d_same(inputs, filters, kernel_size, strides, dilation_rate=1):
    """Strided 2-D convolution with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.

    Note that

       net = conv2d_same(inputs, num_outputs, 3, stride=stride)

    is equivalent to

       net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=1,
       padding='SAME')
       net = subsample(net, factor=stride)

    whereas

       net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=stride,
       padding='SAME')

    is different when the input's height or width is even, which is why we add
    the current function. For more details, see
    ResnetUtilsTest.testConv2DSameEven().

    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      filters: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      strides: An integer, the output strides.
      dilation_rate: An integer, rate for atrous convolution.

    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels]
        with the convolution output.
    """
    if strides == 1:
        return tf.layers.conv2d(
            inputs, filters, kernel_size, strides=1, use_bias=False,
            dilation_rate=dilation_rate, padding='same',
        )
    else:
        kernel_size_effective = (
            kernel_size + (kernel_size - 1) * (dilation_rate - 1)
        )
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )
        return tf.layers.conv2d(
            inputs, filters, kernel_size, strides=strides, use_bias=False,
            dilation_rate=dilation_rate, padding='valid',
        )


def stack_blocks_dense(net, blocks, output_stride=None):
    """Stacks ResNet `Blocks` and controls output feature density.

    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.

    Second, this function allows the user to explicitly control the ResNet
    output_stride, which is the ratio of the input to output spatial
    resolution.  This is useful for dense prediction tasks such as semantic
    segmentation or object detection.

    Most ResNets consist of 4 ResNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive ResNet blocks. This
    results to a nominal ResNet output_stride equal to 8. If we set the
    output_stride to half the nominal network stride (e.g., output_stride=4),
    then we compute responses twice.

    Control of the output feature density is implemented by atrous convolution.

    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      output_stride: If `None`, then the output will be computed at the nominal
        network stride. If output_stride is not `None`, it specifies the
        requested ratio of input to output spatial resolution, which needs to
        be equal to the product of unit strides from the start up to some level
        of the ResNet.  For example, if the ResNet employs units with strides
        1, 2, 1, 3, 4, 1, then valid values for the output_stride are 1, 2, 6,
        24 or None (which is equivalent to output_stride=24).

    Returns:
      net: Output tensor with stride equal to the specified output_stride.
      endpoints: A dictionary from components of the network to the
        corresponding activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever
    # applying the next residual unit would result in the activations having
    # stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    endpoints_collection = {}

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as scope:
            for i, unit in enumerate(block.args):
                if (output_stride is not None
                        and current_stride > output_stride):
                    raise ValueError(
                        'The target output_stride cannot be reached.'
                    )

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need
                    # to employ atrous convolution with stride=1 and multiply
                    # the atrous rate by the current unit's stride for use in
                    # subsequent layers.
                    if (output_stride is not None
                            and current_stride == output_stride):
                        net = block.unit_fn(
                            net, rate=rate, **dict(unit, stride=1)
                        )
                        rate *= unit.get('stride', 1)

                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)

            # Add output of each block to the endpoints collection.
            endpoints_collection[scope.name] = net

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net, endpoints_collection


def resnet_v1(inputs, blocks, training=True, global_pool=True,
              output_stride=None, include_root_block=True, reuse=None,
              scope=None):
    """Generator for v1 ResNet models.

    This function generates a family of ResNet v1 models. See the resnet_v1_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.

    Training for image classification on Imagenet is usually done with [224,
    224] inputs, resulting in [7, 7] feature maps at the output of the last
    ResNet block for the ResNets defined in [1] that have nominal stride equal
    to 32.  However, for dense prediction tasks we advise that one uses inputs
    with spatial dimensions that are multiples of 32 plus 1, e.g., [321,
    321]. In this case the feature maps at the ResNet output will have spatial
    shape [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225,
    225] images results in [8, 8] feature maps at the output of the last ResNet
    block.

    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2]
    all have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features
    at small computational and memory overhead,
    cf. http://arxiv.org/abs/1606.00915.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each
        element is a resnet_utils.Block object describing the units in the
        block.
      training: whether batch_norm layers are in training mode.
      global_pool: If True, we perform global average pooling before computing
        the logits. Set to True for image classification, False for dense
        prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the
        requested ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it.
      reuse: whether or not the network and its variables should be reused. To
        be able to reuse 'scope' must be given.
      scope: Optional variable_scope.

    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out,
        channels_out].  If global_pool is False, then height_out and width_out
        are reduced by a factor of output_stride compared to the respective
        height_in and width_in, else both height_out and width_out equal
        one. `net` is the output of the last ResNet block, potentially after
        global average pooling.
      endpoints: A dictionary from components of the network to the
        corresponding activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse):
        net = inputs

        if include_root_block:
            if output_stride is not None:
                if output_stride % 4 != 0:
                    raise ValueError(
                        'The output_stride needs to be a multiple of 4.'
                    )
                output_stride /= 4

            with tf.variable_scope('conv1'):
                net = conv2d_same(net, 64, 7, strides=2)
                net = tf.layers.batch_normalization(
                    net, momentum=0.997, epsilon=1e-5, training=False,
                    fused=False
                )
                net = tf.nn.relu(net)

            with tf.variable_scope('pool1'):
                net = tf.layers.max_pooling2d(net, [3, 3], strides=2)

        net, endpoints = stack_blocks_dense(net, blocks, output_stride)

        if global_pool:
            # Global average pooling.
            net = tf.reduce_mean(
                net, [1, 2], name='pool5', keepdims=True
            )

        return net, endpoints


resnet_v1.default_image_size = 224


def resnet_v1_101(inputs, training=True, global_pool=True,
                  output_stride=None, reuse=None, scope='resnet_v1_101'):

    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]

    return resnet_v1(
        inputs,
        blocks,
        training,
        global_pool,
        output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope,
    )


def resnet_v1_101_tail(inputs, scope='resnet_v1_101'):
    blocks = [
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]

    return resnet_v1(
        inputs, blocks, global_pool=False, training=False,
        include_root_block=False, scope=scope,
    )
