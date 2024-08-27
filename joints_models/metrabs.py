import tensorflow as tf
import tensorflow_hub as tfhub

model = tfhub.load('https://bit.ly/metrabs_l')
image = tf.image.decode_jpeg(tf.io.read_file('img/test_image_3dpw.jpg'))
pred = model.detect_poses(image)
pred['boxes'], pred['poses2d'], pred['poses3d']