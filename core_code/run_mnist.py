import numpy as np
np.random.seed(2)
# use GPU mem incrementally
import tensorflow as tf
from utils import *
from neurallogic import *
from lenet import *
from data_prepare import *
from intergrated_model import *
from data_gen_digits import *

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)


# set global parameters
plot_epoch = 200

model_dir="C:\\Users\\IlanaHeintz\\neuroplex_nosync"
saved_result_data = os.path.join(model_dir,'result_data/formal_sim_1-validation-mae-modify.pkl')


############################ CE definition ############################
simulation_name = 'digit_gen_test_with_mnist'

NL_model_name = 'NL_' + simulation_name + ".hdf5"
PL_model_name = 'PNL_' + simulation_name


num_event_type = 10   # total num of unique events  3x3 + 1 unknown

ce_fsm_list = [[1,2,3], [4,5,6], [7,8,9], [0,0,0] ]
#ce_timing_list = [[INF, INF], [INF, INF], [INF, INF], [INF, INF]]
#num_attributes = 1
num_logic_samples = 100000
window_size = 10


############################ Training Neural Logic Models ############################
NL_model_path = os.path.join(model_dir, NL_model_name)
train_neurallogic_model(NL_model_path,
                        num_event_type, 
                        ce_fsm_list, 
                        window_size,
                        num_logic_samples,
                        verify_logic=False,
                        diagnose=False)

# Returns a compiled model identical to the previous one
loading_path = os.path.join(model_dir, NL_model_name)
neuralLogic_model = tf.keras.models.load_model(loading_path)
#neuralLogic_model.name="neurallogic"
print('Loading model successfully from ', loading_path)


############################ Preparing all dataset required ############################
num_fulltrain_samples = 10000

#mnist_data_event_path = PL_model_name+'_data_' + str(num_fulltrain_samples) +'.npz'

# Generate new event sequences for "full" training
# and assign labels according to our complex event definitions
ce_regexes = []
ce_names = []
for fsmspec in ce_fsm_list:
    name = "".join([str(i) for i in fsmspec])
    ce_regexes.append(re.compile(name))
    ce_names.append(name)

mnist_xtrain, mnist_ytrain, mnist_xtest, mnist_ytest = load_mnist_data()
mnist_rows_dict = {}
for i in range(10):
    mnist_rows_dict[i] = [k for k in range(len(mnist_ytrain)) if mnist_ytrain[k] == i]

_, fulltrain_labels, fulltrain_image_indices = create_digit_data(num_fulltrain_samples, window_size, ce_regexes, mnist_rows_dict)

# Choose only those rows in the data that correspond to complex events
# If there's no CE in it, there's no label from logic layer, and nothing to learn (?)
ce_indices = (fulltrain_labels.sum(axis=1)!=0)
mnist_ce_only_labels = fulltrain_labels[ce_indices, ]
mnist_ce_image_indices = fulltrain_image_indices[ce_indices, ]
# Rearrange the actual mnist data into the complex event sequences we generated
mnist_train_ce_data = mnist_xtrain[mnist_ce_image_indices]

# modify the testing MNIST dataset for customized LeNet
# (disregard complex events, this is just perception testing)
mnist_x_test, mnist_y_test = modify_mnist_data(mnist_xtest, mnist_ytest, num_event_type)

# split for training and testing    ---# no validation here 
p_train = 0.8
rnd_indices = np.random.rand(mnist_ce_only_labels.shape[0]) < p_train

ce_mnist_data_event_validation = mnist_train_ce_data[~rnd_indices]
ce_data_label_validation = mnist_ce_only_labels[~rnd_indices]

ce_mnist_data_event_train = mnist_train_ce_data[rnd_indices]
ce_data_label_train = mnist_ce_only_labels[rnd_indices]

############################ Training with Proposed ############################
# generate a new LeNet model from scratch
lenetModel = LeNet_mnist(num_output = num_event_type)  # 2 events
#lenetModel.name="lenetModel"
print("Evaluate initialized LeNet perception model")
score = lenetModel.evaluate(mnist_x_test, mnist_y_test, verbose=1)
print('Test loss: %3f,  \t \tTest Accuracy: %4f'%(score[0], score[1]))

# Need to correct the loss function b/c that's what makes this special
# Initialize the model
final_model = intergrated_model(lenetModel, neuralLogic_model, 
                                window_size, num_event_type,
                                omega_value = 1e-4,
                                load_nl_weights = True,
                                nl_trainable = False,
                                #loss = 'combined_loss',  # use semantic loss here
                                loss = 'mse_loss',
                                diagnose = False)

epochs = plot_epoch
verbose = 1

save_path = os.path.join(model_dir, PL_model_name +'.hdf5')
es = tf.keras.callbacks.EarlyStopping(monitor='val_MAE', mode='min', verbose=1, patience=plot_epoch)
mc = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_MAE', mode='min', verbose=False, save_best_only=True)
cb_list = [es, mc]

# Train the model
print("Train proposed model with combined loss (currently MSE) for up to {} epochs".format(epochs))
H = final_model.fit(ce_mnist_data_event_train,
                    ce_data_label_train, 
                    batch_size = 256, 
                    epochs = epochs,
                    verbose=verbose,
                    shuffle=True,
                    callbacks=cb_list,
                    validation_data = (ce_mnist_data_event_validation, ce_data_label_validation))


hist_NL = H

############################ Training with ablated version ############################
# generate a new LeNet model from scratch
lenetModel_2 = LeNet_mnist(num_output = num_event_type)  # 2 events
#lenetModel_2.name="lenetModel_2"
# score = lenetModel_2.evaluate(mnist_x_test, mnist_y_test, verbose=1)
# print('Test loss: %3f,  \t \tTest Accuracy: %4f'%(score[0], score[1]))

# Currently same as above
final_model_ablation = intergrated_model(lenetModel_2, neuralLogic_model, 
                                window_size, num_event_type,
                                load_nl_weights = True,
                                nl_trainable = False,
                                loss = 'mse_loss',
                                diagnose = False)
epochs = plot_epoch

save_path = os.path.join(model_dir, PL_model_name +'_ablation'+'.hdf5')
es = tf.keras.callbacks.EarlyStopping(monitor='val_MAE', mode='min', verbose=1, patience=200)
mc = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_MAE', mode='min', verbose=False, save_best_only=True)
cb_list = [es, mc]

print("Train full network with MSE loss for maximum {} epochs".format(epochs))
H = final_model_ablation.fit( ce_mnist_data_event_train , ce_data_label_train, 
                        batch_size = 256, 
                        epochs = epochs,
                        verbose=verbose,
                        shuffle=True,
                        callbacks=cb_list,
                        validation_data = (ce_mnist_data_event_validation, ce_data_label_validation) )
#                         validation_split = 0.2)

hist_ablation = H

############################ Training with CRNN ############################
# generate a new LeNet model from scratch
lenetModel_3 = LeNet_mnist(num_output = num_event_type)  # 2 events
#lenetModel_3.name="lenetModel_3"
#score = lenetModel_3.evaluate(mnist_x_test, mnist_y_test, verbose=1)
#print('Test loss: %3f,  \t \tTest Accuracy: %4f'%(score[0], score[1]))

# Don't use the pretrained logic LSTM weights, train all from scratch
final_model_scratch = intergrated_model(lenetModel_3, neuralLogic_model, 
                                window_size, num_event_type,
                                load_nl_weights = False,
                                nl_trainable = True,
                                loss = 'mse_loss',
                                diagnose = False)

epochs = plot_epoch

save_path = os.path.join(model_dir, PL_model_name +'_scratch'+'.hdf5')
es = tf.keras.callbacks.EarlyStopping(monitor='val_MAE', mode='min', verbose=1, patience=200)
mc = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_MAE', mode='min', verbose=False, save_best_only=True)
cb_list = [es, mc]

print("Train full pipeline with initialized logic model for maximum {} epochs".format(epochs))
H = final_model_scratch.fit( ce_mnist_data_event_train , ce_data_label_train, 
                        batch_size = 256, 
                        epochs = epochs,
                        verbose=verbose,
                        shuffle=True,
                        callbacks=cb_list,
                        validation_data = (ce_mnist_data_event_validation, ce_data_label_validation) )
#                         validation_split = 0.2)

hist_scratch = H

############################ Training with C3D ############################
"""
def c3d_model(inputdata,input_label):
    print('building the model ... ')            
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=inputdata.shape[1:]))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3' ,dim_ordering="th"))
    # 4th layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool4', dim_ordering="th"))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(1024,  name='fc6'))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(128, name='DENSE_2'))
    model.add(Activation('relu', name = 'ACT_2'))
    #     model.add(Dropout(.5))
    model.add(Dense(128, name='DENSE_1'))
    model.add(Activation('relu', name = 'ACT_1'))

    model.add(Dense(input_label.shape[1], activation='linear', name = 'SOFT'))
    
    # Compile the network
    model.compile(
        loss = "mean_squared_error",
#     loss = "categorical_crossentropy",

#         loss = grad_loss,
    optimizer = SGD(lr = 0.01),
    metrics = ["mae"])
    
    return model

c3d_model = c3d_model(v_mnist_data_event ,v_data_label)

epochs = plot_epoch
diagnose = True

H = c3d_model.fit( v_mnist_data_event ,v_data_label, 
                        batch_size = 256, 
                        epochs = epochs,
                        verbose=1,
                        shuffle=True,
                        validation_data = (v_mnist_data_event_valid, v_data_label_valid) )
#                         validation_split = 0.2
#              )

hist_c3d = H
"""
############################ Comparison between NL and baselines ############################
# evaluation: plot the learning curves
hist1 = hist_NL.history
hist2 = hist_ablation.history
hist3 = hist_scratch.history
#hist4 = hist_c3d.history
print(hist1.keys() )

import matplotlib.pyplot as plt

# MAE figure
fig = plt.figure(figsize=(8, 6))
plt.plot(hist1['val_MAE'], '-', linewidth=3)
plt.plot(hist2['val_MAE'], '-', linewidth=3)
plt.plot(hist3['val_MAE'], '-', linewidth=3)
#plt.plot(hist4['val_mae'], '-', linewidth=3)

plt.legend(['AdaPerception', 'AdaPerception w/o L_semantic', 'Training from scratch', 'C3D model'])
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('filename.png', dpi=600)
plt.show()


lenet_score1 = lenetModel.evaluate(mnist_x_test, mnist_y_test, verbose=0)
# print('Test loss: %3f,  \t \tTest Accuracy: %4f'%(lenet_score1[0], lenet_score1[1]))
print('Proposed acc:\t %4f'%(lenet_score1[1]))
lenet_score2 = lenetModel_2.evaluate(mnist_x_test, mnist_y_test, verbose=0)
print('Ablation acc:\t %4f'%(lenet_score2[1]))
lenet_score3 = lenetModel_3.evaluate(mnist_x_test, mnist_y_test, verbose=0)
print('Scratch acc:\t %4f'%(lenet_score3[1]))

final_train_score1 = final_model.evaluate(ce_mnist_data_event_train, ce_data_label_train, verbose = 0)
# print('Train MSE: %3f,  \t \tTrain MAE: %4f'%(final_train_score1[0], final_train_score1[1]))
final_train_score2 = final_model_ablation.evaluate(ce_mnist_data_event_train, ce_data_label_train, verbose = 0)
final_train_score3 = final_model_scratch.evaluate(ce_mnist_data_event_train, ce_data_label_train, verbose = 0)
#final_train_score4 = c3d_model.evaluate(v_mnist_data_event, v_data_label, verbose = 0)

print('Method \t\t\t MSE \t\t\t MAE \t\t')
print('Proposed \t\t %3f  \t\t  %4f'%(final_train_score1[0], final_train_score1[1]))
print('Ablation \t\t %3f  \t\t  %4f'%(final_train_score2[0], final_train_score2[1]))
print('Scratch \t\t %3f  \t\t  %4f'%(final_train_score3[0], final_train_score3[1]))
#print('C3Dnet \t\t\t %3f  \t\t  %4f'%(final_train_score4[0], final_train_score4[1]))

final_test_score1 = final_model.evaluate(ce_mnist_data_event_validation, ce_data_label_validation, verbose = 0)
# print('Train MSE: %3f,  \t \tTrain MAE: %4f'%(final_test_score1[0], final_test_score1[1]))
final_test_score2 = final_model_ablation.evaluate(ce_mnist_data_event_validation, ce_data_label_validation, verbose = 0)
final_test_score3 = final_model_scratch.evaluate(ce_mnist_data_event_validation, ce_data_label_validation, verbose = 0)
#final_test_score4 = c3d_model.evaluate(v_mnist_data_event_valid, v_data_label_valid, verbose = 0)

print('Method \t\t MSE \t \t MAE \t\t')
print('Proposed \t\t %3f  \t\t  %4f'%(final_test_score1[0], final_test_score1[1]))
print('Ablation \t\t %3f  \t\t  %4f'%(final_test_score2[0], final_test_score2[1]))
print('Scratch \t\t %3f  \t\t  %4f'%(final_test_score3[0], final_test_score3[1]))
#print('C3Dnet \t\t %3f  \t\t  %4f'%(final_test_score4[0], final_test_score4[1]))

def ce_model_acc(v_mnist_data_event, v_data_label, model):
    pred_score = model.predict(v_mnist_data_event)
    accuracy = sum(v_data_label == pred_score.round() ) / v_data_label.shape[0]
    avg_acc = accuracy.mean()
    return accuracy, avg_acc

_, final_train_acc1 = ce_model_acc(ce_mnist_data_event_validation, ce_data_label_validation, final_model)
_, final_train_acc2 = ce_model_acc(ce_mnist_data_event_validation, ce_data_label_validation, final_model_ablation)
_, final_train_acc3 = ce_model_acc(ce_mnist_data_event_validation, ce_data_label_validation, final_model_scratch)
#_, final_train_acc4 = ce_model_acc(v_mnist_data_event_valid, v_data_label_valid, c3d_model)

print("Proposed \t Ablation \t Scratch \t C3D \t")
#print('%3f \t %3f \t %3f \t %3f \t '%(final_train_acc1, final_train_acc2, final_train_acc3, final_train_acc4) )
print('%3f \t %3f \t %3f \t  '%(final_train_acc1, final_train_acc2, final_train_acc3) )


############################ saving results ############################
#hist_list = [hist1, hist2, hist3, hist4]
hist_list = [hist1, hist2, hist3]
lenet_list = [lenet_score1, lenet_score2, lenet_score3]
#train_score_list = [final_train_score1, final_train_score2, final_train_score3, final_train_score4]
train_score_list = [final_train_score1, final_train_score2, final_train_score3]
#test_score_list = [final_test_score1, final_test_score2, final_test_score3, final_test_score4]
test_score_list = [final_test_score1, final_test_score2, final_test_score3]
#acc_list = [final_train_acc1, final_train_acc2, final_train_acc3, final_train_acc4]
acc_list = [final_train_acc1, final_train_acc2, final_train_acc3]

saving_dict = {'hist':hist_list, 
              'lenet_acc':lenet_list, 
              'train_score':train_score_list, 
              'test_score':test_score_list, 
              'train_acc':acc_list, }


import pickle
with open(saved_result_data, 'wb') as f:
    pickle.dump(saving_dict, f)
    
# check loaded data
with open(saved_result_data, 'rb') as f:
    mynewdict = pickle.load(f)

mynewdict.keys()



