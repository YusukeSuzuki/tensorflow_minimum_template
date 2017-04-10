import tensorflow as tf

def read_image_op(filename_queue, reader, height, width):
    _, raw = reader.read(filename_queue)

    read_image = tf.image.decode_jpeg(raw, channels=3)
    read_image = tf.to_float(read_image) / 255.
    read_image = tf.image.resize_images(read_image, [height, width])
    read_image = tf.image.random_flip_left_right(read_image)
    return read_image

