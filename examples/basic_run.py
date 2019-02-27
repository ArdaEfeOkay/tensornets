import sys
sys.path.append("..") 

import tensorflow as tf
import tensornets as nets

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
model = nets.VGG16(inputs)

assert isinstance(model, tf.Tensor)

images = ['cat.png', 'dog.jpg', 'eagle.jpg', 'giraffe.jpg', 'horses.jpg']

img = [nets.utils.load_img(image, target_size=256, crop_size=224) for image in images]

#assert img.shape == (1, 224, 224, 3)

with tf.Session() as sess:
    print("**************************")
    print("**************************")
    model.print_summary()
    print("**************************")
    print("**************************")
    sess.run(model.pretrained())

    for  _img in img:
        _img = model.preprocess(_img)  # equivalent to img = nets.preprocess(model, img)
          # equivalent to nets.pretrained(model)
        preds = sess.run(model, {inputs: _img})


        print(nets.utils.decode_predictions(preds, top=2)[0])



