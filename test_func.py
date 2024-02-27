import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import image_to_train as tr


def downsample(path, scale, method='bicubic', batch=False, input_channels=3):
    assert 0 < scale < 1, 'downsample scale must be between 0 and 1'
    im = tf.io.read_file(path)
    im = tf.image.decode_image(im, channels=input_channels)
    im = tf.image.convert_image_dtype(im, dtype=tf.float32)
    im_shape = im.shape[:2]
    if im.shape[-1] == 3:
        im = tr.to_luminance(im)
    im = tr.resize_image(im, scale, method)
    im = tf.image.resize(im, im_shape, method=method)
    im = tf.squeeze(im)
    if batch:
        im = tf.expand_dims(im, axis=0)
    return im


def upsample(path, scale, method='bicubic', batch=False, input_channels=3):
    assert scale >= 1, 'downsample scale must be greater than 1'
    im = tf.io.read_file(path)
    im = tf.image.decode_image(im, channels=input_channels)
    im = tf.image.convert_image_dtype(im, dtype=tf.float32)
    if im.shape[-1] == 3:
        im = tr.to_luminance(im)
    if scale != 1:
        im = tr.resize_image(im, scale, method)
    im = tf.squeeze(im)
    if batch:
        im = tf.expand_dims(im, axis=0)
    return im


def write_images_to_path(path, images, file_names=[]):
    encoded_images = []
    if not os.path.exists(path):
        os.mkdir(path)
    for im in images:
        if not tf.dtypes.as_dtype(im.dtype).is_integer:
            tf.image.convert_image_dtype(im, tf.uint8)
        encoded_image = tf.image.encode_png(im)
        encoded_images.append(encoded_image)
    if file_names:
        for im, fname in zip(encoded_images, file_names):
            im_path = os.path.join(path, fname+'.png')
            tf.io.write_file(im_path, im)
    else:
        for i, im in enumerate(encoded_images):
            tf.io.write_file(os.path.join(path, str(i)+'.png'), im)


def luminance_to_rgb(lum_img, rgb_img):
    if tf.dtypes.as_dtype(lum_img.dtype).is_integer:
        lum_img = tf.image.convert_image_dtype(lum_img, tf.float32)
    if tf.dtypes.as_dtype(rgb_img.dtype).is_integer:
        rgb_img = tf.image.convert_image_dtype(rgb_img, tf.float32)
    if lum_img.shape != rgb_img.shape:
        rgb_img = tf.image.resize(rgb_img, size=lum_img.shape[:2])
    yuv_image = tf.image.rgb_to_yuv(rgb_img)
    yuv_image = tf.concat([tf.expand_dims(lum_img, axis=-1), yuv_image[:, :, 1:]], axis=-1)
    recon_rgb = tf.image.yuv_to_rgb(yuv_image)
    recon_rgb = tf.image.convert_image_dtype(recon_rgb, tf.uint8)
    return recon_rgb


def load_x2_from_weights(model, weight_path):
    """
    function for loading the weights provided by the authors of DWSR
    example for format of path to weight:
        x2_weight_path = os.path.join('Weightx2', 'x2.ckpt')
    """

    ckpt_reader = tf.train.load_checkpoint(weight_path)
    tensor_names = ckpt_reader.get_variable_to_shape_map().keys()
    """
    names of the layers are as follows:
    conv_00_w, conv_00_b
    conv_??_w, conv_??_b
    conv_20_w, conv_20_b
    """
    names_list = []
    full_names_list = []
    for tensor_name in tensor_names:
        # Read the tensor value
        # tensor_value = ckpt_reader.get_tensor(tensor_name)
        # print(f"Layer: {tensor_name}, Shape: {tensor_value.shape}")
        if tensor_name.endswith('w') or tensor_name.endswith('b'):
            names_list.append(tensor_name)
        full_names_list.append(tensor_name)

    names_list.sort()
    names_list = np.array(names_list)
    grouped_names_list = []
    for x in range(0, len(names_list), 2):
        grouped_names_list.append([names_list[x], names_list[x + 1]])
    grouped_names_list = np.array(grouped_names_list)

    for i, (b_name, w_name) in enumerate(grouped_names_list):
        b_tensor = ckpt_reader.get_tensor(b_name)
        w_tensor = ckpt_reader.get_tensor(w_name)

        model.layers[i].set_weights([w_tensor, b_tensor])
    return model


def calculate_ssim(X, Y):
    ssim_scores = []
    for x, y in zip(X, Y):
        ssim_scores.append(tf.image.ssim(x, y, max_val=1.0))
    avg_ssim_score = sum(ssim_scores) / len(ssim_scores)
    return avg_ssim_score


def calculate_psnr(X, Y):
    psnr_scores = []
    for x, y in zip(X, Y):
        psnr_scores.append(tf.image.psnr(x, y, max_val=1.0))
    avg_psnr_score = sum(psnr_scores) / len(psnr_scores)
    return avg_psnr_score
