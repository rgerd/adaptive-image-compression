import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras.utils as ku

from keras.datasets import cifar100 # https://keras.io/datasets/
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = ku.to_categorical(y_train, 100)
y_test = ku.to_categorical(y_test, 100)

images = None

input_shape = x_train[0].shape

# Feature extraction
x_1_input = kl.Input(shape=input_shape)
coeffs = []
downscales = []
resamples = []
output = None

NUM_FEATURE_SCALES = 2
for i in range(NUM_FEATURE_SCALES):
    if i == 0:
        scale_input = x_1_input
    else:
        scale_input = downscales[i - 1]

    F_shape = (3, 3) # if i < 3 else (1, 1)
    F = kl.LeakyReLU(alpha=0.2)(kl.Conv2D(64, F_shape, strides=(1, 1))(scale_input))
    
    # Downsample for multiscale feature extraction
    D = kl.Conv2D(input_shape[2], (4, 4), strides=(2, 2), padding='same')(scale_input)

    # G_m(.)
    if i == 0:
        G = kl.Conv2D(32, (3, 3), strides=(1, 1))(F)
    else:
        downscale_amount = 2 ** i
        # Upsample for the sum to G(.)
        # https://towardsdatascience.com/transpose-convolution-77818e55a123
        G = kl.Conv2DTranspose(32, (2, 2), strides=(downscale_amount, downscale_amount))(F)

    coeffs.append(F)
    downscales.append(D)
    resamples.append(G)

merged_resample = kl.Add()(resamples)
output = kl.LeakyReLU(alpha=0.2)(kl.Conv2D(48, (3, 3), padding='same')(merged_resample))

extraConv1 = kl.Conv2D(32, (3, 3), activation='relu')(output)
extraConv2 = kl.Conv2D(16, (3, 3), activation='relu')(extraConv1)
cifar_output = kl.Dense(100, activation='softmax')(kl.Flatten()(extraConv2))

if __name__ == '__main__':
    model = km.Model(inputs=x_1_input, outputs=cifar_output)
    model.compile(optimizer=ko.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(
        x_train, y_train,
        batch_size=16, epochs=10,
        validation_data=(x_test, y_test))