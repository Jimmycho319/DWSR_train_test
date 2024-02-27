import tensorflow as tf
import numpy as np
import os
import pywt as pw
import matplotlib.pyplot as plt
import time

"""
Preprocessing functions used in training and analysis
"""


def display_image(image, figsize=(5,5)):
    if not isinstance(image, np.ndarray):
        image = image.numpy()
    image = tf.squeeze(image)
    if image.shape[-1] == 4:
        image = np.moveaxis(image, -1, 0)
    if len(image) == 4:
        LL, LH, HL, HH = image
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs[0, 0].imshow(LL, cmap='gray')
        axs[0, 0].set_title('(LL)', fontsize=14)
        axs[0, 0].axis('off')

        axs[0, 1].imshow(LH, cmap='gray')
        axs[0, 1].set_title('(LH)', fontsize=14)
        axs[0, 1].axis('off')

        axs[1, 0].imshow(HL, cmap='gray')
        axs[1, 0].set_title('(HL)', fontsize=14)
        axs[1, 0].axis('off')

        axs[1, 1].imshow(HH, cmap='gray')
        axs[1, 1].set_title('(HH)', fontsize=14)
        axs[1, 1].axis('off')
        plt.show()
    else:
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()


def to_luminance(image):
    assert image.dtype == tf.float32, 'image must be converted to float32 format'
    assert (len(image.shape)==2
            or (len(image.shape)==3 and image.shape[2] in (1, 3))), "image must be in luminance or rgb format"
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
    if image.shape[2] == 3:
        image = tf.image.rgb_to_grayscale(image)
    return image


def resize_image(image, scale, scale_method='bicubic'):
    # to add: check if the image format is channel first or channel last
    # if tf.channel == 'channel-last':

    assert len(image.shape) == 3, "incorrect image format"
    resize_shape = [int(image.shape[0]*scale), int(image.shape[1]*scale)]  # assuming channel-last image format
    resized_image = tf.image.resize(image, resize_shape, method=scale_method)  # do we need to specify to keep the aspect ratio?
    if tf.reduce_max(image) > 1:
        return tf.cast(resized_image, tf.uint8)
    return resized_image


def dwt_transform(image, wv='db1', channel_last=False):
    image = tf.squeeze(image)
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = to_luminance(image)
        image = tf.squeeze(image)

    assert len(image.shape) == 2, 'image must be 2d array'
    t_coeffs = pw.dwt2(image, wv)
    a, (h, v, d) = t_coeffs
    t_coeffs = np.array([a, h, v, d])
    if channel_last:
        t_coeffs = tf.transpose(t_coeffs, [1, 2, 0])
    return t_coeffs


def bands_to_image(sub_bands, wv='db1'):
    """
    idwt_transform equivalent function
    """
    sub_bands = tf.squeeze(sub_bands)
    if isinstance(sub_bands, type(tf.constant(0))) or isinstance(sub_bands, type(np.array([]))):
        if sub_bands.shape[-1] == 4 and len(sub_bands.shape) == 3:
            sub_bands = tf.transpose(sub_bands, [2,0,1])
    if len(sub_bands) == 4:
        sub_bands = (sub_bands[0], (sub_bands[1], sub_bands[2], sub_bands[3]))
    return pw.idwt2(sub_bands, wv)


def preprocess_single_image(image, wv='db1', scale=2, scale_method='bicubic', batch_dimension=True, channel_last=False):
    """
    use when testing a single image or batch of images
    takes rgb image and converts image to luminance sub-bands
    for training, use the write_preprocessed_images() to generate a dataset for training
    :return: single group of sub-bands (image of size (1xNxNx4)
    """
    # image = image.copy()
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = to_luminance(image)
    if type(scale) is not int and type(scale) is not float:
        image = tf.image.resize(image, scale, method=scale_method)
    elif scale != 1:
        image = resize_image(image, scale, scale_method)
    image = tf.squeeze(image)
    t_coeffs = dwt_transform(image, wv)
    if channel_last:
        t_coeffs = np.moveaxis(t_coeffs, 0, -1)
    if batch_dimension:
        t_coeffs = t_coeffs[np.newaxis, ...]        # create batch axis for feeding into model
    return t_coeffs


def preprocess_single_train(image, wv='db1', downsample_scale=0.5, scale_method='bicubic'):
    """
    for use in evaluating the accuracy of the model
    use this when evaluating the PSNR or SSIM
    downsample_scale must be between 0 and 1
    :return: pair of sub-bands (two images of size (NxNx4))
    """

    assert 1 > downsample_scale > 0, "downsample_scale must be between 0 and 1"

    # image = image.copy()
    original_shape = [image.shape[0], image.shape[1]]
    HRSB = preprocess_single_image(tf.identity(image), wv=wv, scale=1, batch_dimension=True)

    # review whether to downsample first or to change to grayscale first
    downscaled_image = resize_image(tf.identity(image), scale=downsample_scale, scale_method=scale_method)
    LRSB = preprocess_single_image(downscaled_image, wv=wv, scale=original_shape, batch_dimension=True)

    y = HRSB - LRSB                             # attain delta SB
    return tf.concat([LRSB, y], axis=0)


def image_to_subimage(image, subimage_size=[41,41], overlap=10):
    assert len(image.shape) == 3, 'image must not be batched'
    channel_axis = np.argmin(image.shape)
    vert, hor = subimage_size[0], subimage_size[1]
    subimages = []
    if channel_axis == 2:
        height, width, _ = image.shape
        vert_idx = [0] + [x for x in range(vert-overlap, height-vert, vert-overlap)]
        hor_idx  = [0] + [x for x in range(hor-overlap, width-hor, hor-overlap)]
        for i in vert_idx:
            for j in hor_idx:
                subimages.append(image[i:i+vert, j:j+hor])
    elif channel_axis == 0:
        _, height, width = image.shape
        vert_idx = [0] + [x for x in range(vert-overlap, height-vert, vert-overlap)]
        hor_idx  = [0] + [x for x in range(hor-overlap, width-hor, hor-overlap)]
        for i in vert_idx:
            for j in hor_idx:
                subimages.append(image[:, i:i+vert, j:j+hor])
    else:
        raise Exception
    return np.array(subimages)


def preprocess_and_save_train_from_dir(input_directory, output_directory, wv='db1', downsample_scale=0.5,
                                       scale_method='bicubic', subimage=[41,41], overlap=10):
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    for file in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file)
        im = tf.io.read_file(file_path)
        im = tf.image.decode_image(im)
        im = preprocess_single_train(im, wv=wv, downsample_scale=downsample_scale,
                                     scale_method=scale_method)
        filename = os.path.splitext(file)[0]

        if subimage:
            if type(subimage) == int:
                subimage = [subimage, subimage]
            x_sub = image_to_subimage(im[0], subimage, overlap)
            y_sub = image_to_subimage(im[1], subimage, overlap)
            x_out = os.path.join(output_directory, filename+'x')
            y_out = os.path.join(output_directory, filename+'y')

            np.save(x_out, x_sub)
            np.save(y_out, y_sub)

        else:
            out = os.path.join(output_directory, filename)
            np.save(out, im)
    print('{} files successfully preprocessed and written to {}'.format(
        len(os.listdir(input_directory)), output_directory))


def preprocess_and_save_tfrecord_from_dir(input_directory, output_directory, wv='db1', downsample_scale=0.5,
                                       scale_method='bicubic', subimage=[41,41], overlap=10):
    """saves files as tfrecords"""
    def _array_to_example(x, y):
        x_serialized = tf.io.serialize_tensor(x)
        y_serialized = tf.io.serialize_tensor(y)
        feature = {
            'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_serialized.numpy()])),
            'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_serialized.numpy()]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
        os.mkdir(os.path.join(output_directory, 'train'))
        os.mkdir(os.path.join(output_directory, 'validation'))
    if type(subimage) == int:
        subimage = [subimage, subimage]

    train_array_list = []
    val_array_list = []
    for idx, file in enumerate(os.listdir(input_directory)):
        file_path = os.path.join(input_directory, file)
        im = tf.io.read_file(file_path)
        im = tf.image.decode_image(im)
        im = preprocess_single_train(im, wv=wv, downsample_scale=downsample_scale,
                                     scale_method=scale_method)

        x_sub = image_to_subimage(im[0], subimage, overlap)
        y_sub = image_to_subimage(im[1], subimage, overlap)

        if idx % 5 == 0:
            val_array_list += [[single_x, single_y] for single_x, single_y in zip(x_sub, y_sub)]
        else:
            train_array_list += [[single_x, single_y] for single_x, single_y in zip(x_sub, y_sub)]

        if idx != 0 and idx % 50 == 0:
            start = time.time()
            with tf.io.TFRecordWriter(os.path.join(output_directory, 'train', str(idx)+'.tfrecord')) as writer:
                for x, y in train_array_list:
                    example = _array_to_example(x, y)
                    writer.write(example)
            end = time.time()
            print(end-start)
            train_array_list = []
        if idx != 0 and idx % 250 == 0:
            with tf.io.TFRecordWriter(os.path.join(output_directory, 'validation', str(idx)+'.tfrecord')) as writer:
                for x, y in val_array_list:
                    example = _array_to_example(x, y)
                    writer.write(example)
            val_array_list = []

    if train_array_list:
        with tf.io.TFRecordWriter(os.path.join(output_directory, 'train', 'final' + '.tfrecord')) as writer:
            for x, y in train_array_list:
                example = _array_to_example(x, y)
                writer.write(example)
    if val_array_list:
        with tf.io.TFRecordWriter(os.path.join(output_directory, 'validation', 'final' + '.tfrecord')) as writer:
            for x, y in val_array_list:
                example = _array_to_example(x, y)
                writer.write(example)

    print('{} files successfully preprocessed and written to {}'.format(
        len(os.listdir(input_directory)), output_directory))


def read_image_from_tf(filename):
    def _parse_function(example_proto):
        features = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_feature = tf.io.parse_single_example(example_proto, features)
        x = tf.io.parse_tensor(parsed_feature['x'], out_type=tf.float32)
        y = tf.io.parse_tensor(parsed_feature['y'], out_type=tf.float32)
        return x, y

    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset



def unpack_numpy_subimages(path):
    """
    from path to directory, by following the format we used to save the subimages sub bands to the directory,
    recover the subimages as X and Y
    """
    files = sorted(os.listdir(path))
    idx = 0
    failed_counter = 0
    X = []
    Y = []
    while idx < len(files)-1:
        x_filename = os.path.splitext(files[idx])[0]
        y_filename = os.path.splitext(files[idx+1])[0]
        if x_filename[-1] == 'x' and y_filename[-1] == 'y' and x_filename[:-1] == y_filename[:-1]:
            x_numpy = np.load(os.path.join(path, files[idx]))
            y_numpy = np.load(os.path.join(path, files[idx+1]))
            assert len(x_numpy) == len(y_numpy), 'x and y files do not contain the same number of subimages'
            for i in range(len(x_numpy)):
                X.append(x_numpy[i])
                Y.append(y_numpy[i])
            idx += 2
            # print(idx)
        else:
            idx += 1
            failed_counter += 1
            continue

    print('{}/{} files extracted successfully'.format(len(files)-failed_counter, len(files)))
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    # preprocess_and_save_train_from_dir('unfinx2', 'x2_train_subimages_unfin')
    preprocess_and_save_tfrecord_from_dir(input_directory='DIV2K_train_HR', output_directory='x2sub_tfrecord')

'''potential error: trying to compare reduce_max() of a float with the integer 1?'''