import numpy as np
import tensorflow as tf
import keras
import cv2
from glob import glob

def main():
    x, y = [], []

    for i, category_path in enumerate(glob("data/*")):
        for img_path in glob(category_path + '/*'):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            x.append(img.reshape(128, 128, 3))
            y.append(i)

    x = np.array(x)
    y = np.array(y)

    # (128, 128, 3) 이미지를 입력받아서 94개로 분류하는 모델을 만들어주세요. 드롭아웃, L2도 포함해주고 배치 정규화를 Conv2D 레이어 뒤에 추가해주세요. 배치 정규화 뒤에 PRelu를 추가해주세요.
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU())
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(94, activation='softmax'))

    model.summary()
    # 모델을 컴파일해주세요. optimizer는 Adam을 사용하고, loss는 sparse_categorical_crossentropy를 사용해주세요. learning rate는 0.01로 해주세요.
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 모델을 학습해주세요. epoch는 10으로 해주세요. batch size는 32로 해주세요.
    model.fit(x, y, epochs=10, batch_size=32)
    # 모델을 평가해주세요.
    model.evaluate(x, y)

if __name__ == "__main__":
    main()
