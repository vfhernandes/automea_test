
def model_spikes30(X_train, y_train, X_val, y_val): #must be x_train_con, t_train_con etc....
              
    model = tf.keras.Sequential()  

    d1, d2 = 128, 64
    
    k1, k2 = 3, 5
              
    model.add(tf.keras.layers.Conv1D(d1, kernel_size= k1, activation='relu',padding='same', input_shape=(X_train.shape[1],1)))

    model.add(tf.keras.layers.Conv1D(d2, kernel_size= k2, activation='relu',padding='same'))
              
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2,strides=1)) 
              
    dr1 = 0.332
    
    model.add(tf.keras.layers.Dropout(dr1))

    dr2 = 0.947

    model.add(tf.keras.layers.Dropout(dr2))
                            
    model.add(tf.keras.layers.Flatten())
    
    d4 = 64

    model.add(tf.keras.layers.Dense(d4, activation='relu'))
            
    dr3 = 0.125
    
    model.add(tf.keras.layers.Dropout(dr3))
              
    model.add(tf.keras.layers.Dense(3, activation='relu'))

    lrate = 10**-6
              
    rmsprop = tf.keras.optimizers.RMSprop(lr= lrate)
   
    optim = rmsprop
              
    bs = 2
    
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'],
                  optimizer=optim)   
              
    n_par = model.count_params()

    history = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=50,
              verbose=2,
              validation_data=(X_val, y_val))
              

    model.save('model_spikes30.h5')
    history_dict = history.history
    json.dump(history_dict, open('history_spikes30.json', 'w'))
        
        
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Val accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



def model_signal30(X_train, y_train, X_val, y_val): #must be x_train_con, t_train_con etc....
    
    reduction_value = 30
    
    model = tf.keras.Sequential()
    
    d1, d2 = 128, 64
    
    k1, k2 = 7, 7
              
    model.add(tf.keras.layers.Conv1D(d1, kernel_size= k1, activation='relu',padding='same', input_shape=(X_train.shape[1],1)))

    model.add(tf.keras.layers.Conv1D(d2, kernel_size= k2, activation='relu',padding='same'))
                        
    dr1 = 0.199

    model.add(tf.keras.layers.Dropout(dr1))
          
    d3 = 16
    
    k3 = 9
    
    model.add(tf.keras.layers.Conv1D(d3, kernel_size= k3, activation='relu',padding='same'))

    dr2 = 0.001

    model.add(tf.keras.layers.Dropout(dr2))          
              
    model.add(tf.keras.layers.Flatten())
    
    d4 = 32
    
    model.add(tf.keras.layers.Dense(d4, activation='relu'))
              
    dr3 = 0.620

    model.add(tf.keras.layers.Dropout(dr3))
              
    model.add(tf.keras.layers.Dense(3, activation='relu'))
    
    lrate = 10**-4
              
    sgd = tf.keras.optimizers.SGD(lr= lrate)
   
    optim = sgd
              
    bs = 4
    
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'],
                  optimizer=optim)   
              
    n_par = model.count_params()

    history = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=50,
              verbose=2,
              validation_data=(X_val, y_val))
              

    model.save('model_signal30.h5')
    history_dict = history.history
    json.dump(history_dict, open('history_signal30.json', 'w'))
        
        
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Val accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



def model_signal100(X_train, y_train, X_val, y_val): #must be x_train_con, t_train_con etc....
    
    reduction_value = 100
    
    model = tf.keras.Sequential()
    
    d1, d2 = 128, 64
    
    k1, k2 = 7, 7
              
    model.add(tf.keras.layers.Conv1D(d1, kernel_size= k1, activation='relu',padding='same', input_shape=(X_train.shape[1],1)))

    model.add(tf.keras.layers.Conv1D(d2, kernel_size= k2, activation='relu',padding='same'))
                        
    dr1 = 0.199

    model.add(tf.keras.layers.Dropout(dr1))

    n_layers = {{choice(['two', 'three'])}}
              
    d3 = 16
    k3 = 9

    model.add(tf.keras.layers.Conv1D(d3, kernel_size= k3, activation='relu',padding='same'))
   
    dr2 = 0.001

    model.add(tf.keras.layers.Dropout(dr2))
                        
    model.add(tf.keras.layers.Flatten())
    
    d4 = 32
    
    model.add(tf.keras.layers.Dense(d4, activation='relu'))
              
    dr3 = 0.620
    
    model.add(tf.keras.layers.Dropout(dr3))
              
    model.add(tf.keras.layers.Dense(3, activation='relu'))
    
    lrate = 10**-4
              
    sgd = tf.keras.optimizers.SGD(lr= lrate)
   
    optim = sgd
              
    bs = 4
    
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'],
                  optimizer=optim)   
              
    n_par = model.count_params()

    history = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=50,
              verbose=2,
              validation_data=(X_val, y_val))
              

    model.save('model_signal100.h5')
    history_dict = history.history
    json.dump(history_dict, open('history_signal100.json', 'w'))
        
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Val accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}





