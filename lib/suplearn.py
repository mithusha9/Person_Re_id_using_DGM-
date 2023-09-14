import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import mat73
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import numpy as np

# Define the CNN architecture
def build_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x=Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

normalizing_const=0.06

def evaluate_mAP(y_true, y_pred):
    aps = []
    for i in range(y_true.shape[1]):
        y_true_i = y_true[:, i]*normalizing_const
        y_pred_i = y_pred[:, i]*normalizing_const
        threshold = 0.5
        y_true_discrete = (y_true_i > threshold).astype(int)
        y_pred_discrete = (y_pred_i > threshold).astype(int)
        ap_i = average_precision_score(y_true_discrete, y_pred_discrete)
        aps.append(ap_i)
    return np.mean(aps)
# Define the triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

# Load the dataset
mat1 = mat73.loadmat('data/ilids_CNNFeat.mat')
data = mat1['CNNFeat']
input_data = data.reshape(150, 600, 40, 150)
labels1 = scipy.io.loadmat('data/ilids_GraphCost.mat')
labels = labels1['Graph_Cost']
X_train, X_val, y_train, y_val = train_test_split(input_data, labels, test_size=0.2, random_state=42)

# Build the model
input_shape = X_train[0].shape
model = build_model(input_shape)
model.compile(loss=triplet_loss, optimizer=Adam(0.0001))

# Train the model
batch_size = 32
epochs = 10
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

# Evaluate the model
mat2 = mat73.loadmat('data/ilids_CNNTestFeat.mat')
test_data = mat2['CNNTestFeat']
test_data = test_data.reshape(150, 600, 40, 150)
labels2 = scipy.io.loadmat('data/ilids_GraphCost1.mat')
test_labels = labels2['Graph_Cost1']
y_pred = model.predict(test_data)
mAP = evaluate_mAP(test_labels, y_pred) # custom function to compute mAP
print('mAP: {:.4f}'.format(mAP))


