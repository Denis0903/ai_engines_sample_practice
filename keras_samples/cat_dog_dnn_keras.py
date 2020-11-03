import argparse
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os
from mydnn_keras import MyDNN
 
input_size_x = 224
input_size_y = 224
batch_size = 20
input_size_c = 3
output_size = 2
epochs = 10
checkpoint_dir = 'ckpt'
 
parser = argparse.ArgumentParser()
parser.add_argument("--infer", action="store", nargs="?", type=str,
                    help="学習は行わず、このオプションで指定した画像ファイルの判定処理を行う")
args = parser.parse_args()
 
model = MyDNN(input_shape=(input_size_x, input_size_y, input_size_c),
              output_size=output_size)
 
if not args.infer:
    print("学習モード")
    # 学習データの読み込み
    keras_idg = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = keras_idg.flow_from_directory('data/train',
                          target_size=(input_size_x, input_size_y),
                          batch_size=batch_size,
                          class_mode='categorical')
    valid_generator = keras_idg.flow_from_directory('data/valid',
                          target_size=(input_size_x, input_size_y),
                          batch_size=batch_size,
                          class_mode='categorical')
 
  
    # 学習の実行
    num_data_train_dog = len(os.listdir('data/train/dog'))
    num_data_train_cat = len(os.listdir('data/train/cat'))
    num_data_train = num_data_train_dog + num_data_train_cat
 
    num_data_valid_dog = len(os.listdir('data/valid/dog'))
    num_data_valid_cat = len(os.listdir('data/valid/cat'))
    num_data_valid = num_data_valid_dog + num_data_valid_cat
 
    model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        epochs=epochs,
                        steps_per_epoch=num_data_train/batch_size,
                        validation_steps= num_data_valid/batch_size)
 
    # 学習済みモデルの保存
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_weights(os.path.join(checkpoint_dir, 'mydnn_weights.hdf5'))
else:
    print("判定モード")
    # 判定する画像の読み込み
    image_infer = load_img(args.infer, target_size=(input_size_x, input_size_y))
    data_infer = img_to_array(image_infer)
    data_infer = np.expand_dims(data_infer, axis=0)
    data_infer = data_infer / 255.0
  
    # 判定処理の実行
    model.load_weights(os.path.join(checkpoint_dir, 'mydnn_weights.hdf5'))
    result = model.predict(data_infer)[0] * 100
 
    # 判定結果の出力
    if result[0] > result[1]:
      print('Cat (%.1f%%)' % result[0])
    else:
      print('Dog (%.1f%%)' % result[1])
