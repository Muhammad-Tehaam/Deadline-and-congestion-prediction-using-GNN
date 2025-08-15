from __future__ import print_function
import tensorflow as tf
from tensorflow._api.v2.compat.v1 import float32


class Experimental_RouteNet_Model(tf.keras.Model):
    """ Init method for the custom model.

    Args:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        output_units (int): Output units for the last readout's layer.

    Attributes:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        link_update (GRUCell): Link GRU Cell used in the Message Passing step.
        path_update (GRUCell): Path GRU Cell used in the Message Passing step.
        readout (Keras Model): Readout Neural Network. It expects as input the
                               path states and outputs the per-path delay.
    """

    def __init__(self, config, output_units=1):
        super(Experimental_RouteNet_Model, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.link_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.path_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['path_state_dim']))

        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['path_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  ),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  ),
            tf.keras.layers.Dense(output_units,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2_2']))
                                  )
        ])
        self.congestion_prediction = tf.keras.Sequential([
            tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['link_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS_congestion']['readout_units']),
                                  activation=tf.nn.relu,
                                  ),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS_congestion']['readout_units']),
                                  activation=tf.nn.relu,
                                  ),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS_congestion']['readout_units']),
                                  activation=tf.nn.sigmoid,
                                  ),
            tf.keras.layers.Dense(output_units,activation=tf.nn.sigmoid)
        ])


        self.model_2 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['path_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  ),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  ),
            tf.keras.layers.Dense(output_units,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2_2']))
                                  )
        ])
        self.deadline_detection = tf.keras.Sequential([
            tf.keras.layers.Input(shape=128),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  ),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.selu,
                                  ),
            tf.keras.layers.Dense(output_units,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      float(self.config['HYPERPARAMETERS']['l2_2']))
                                  )
        ])






    def call(self, inputs, training=False):
        """This function is execution each time the model is called

        Args:
            inputs (dict): Features used to make the predictions.
            training (bool): Whether the model is training or not. If False, the
                             model does not update the weights.

        Returns:
            tensor: A tensor containing the per-path delay.
        """

        f_ = inputs
        print(" I am in CALL Funciton!!!!!")
        links = f_['links']
        paths = f_['paths']
        # weights = f_['weights']
        seqs = f_['sequences']

        # Compute the shape for the  all-zero tensor for link_state
        shape = tf.stack([
            f_['n_links'],
            int(self.config['HYPERPARAMETERS']['link_state_dim']) - 3
        ], axis=0)
        # Initialize the initial hidden state for links
        link_state = tf.concat([
            tf.expand_dims(f_['link_capacity'], axis=1),
            tf.expand_dims(f_['tx_policies'], axis=1),
            tf.expand_dims(f_['tx_weights'], axis=1),
           # tf.expand_dims(f_['deadline'], axis=1),
           # tf.expand_dims(f_['volume'], axis=1),
           # tf.expand_dims(f_['duration'], axis=1),
           # tf.expand_dims(f_['arrival'], axis=1),

            tf.zeros(shape)
        ], axis=1)

        # Compute the shape for the  all-zero tensor for path_state
        shape = tf.stack([
            f_['n_paths'],
            int(self.config['HYPERPARAMETERS']['path_state_dim']) - 4
        ], axis=0)

        # Initialize the initial hidden state for paths
        path_state = tf.concat([
            tf.expand_dims(f_['bandwith'], axis=1),
            tf.expand_dims(f_['tos'], axis=1),
            tf.expand_dims(f_['packets'], axis=1),
            tf.expand_dims(f_['AvgPkS'], axis=1),
            tf.zeros(shape)
        ], axis=1)



        #congestion = self.congestion_prediction(link_state, training=training)
        # Iterate t times doing the message passing
        for _ in range(int(self.config['HYPERPARAMETERS']['t'])):

            # The following lines generate a tensor of dimensions [n_paths, max_len_path, dimension_link] with all 0
            # but the link hidden states
            h_tild = tf.gather(link_state, links)

            #ids = tf.stack([paths, seqs, weights], axis=1)
            ids = tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs) + 1
            shape = tf.stack([
                f_['n_paths'],
                max_len,
                int(self.config['HYPERPARAMETERS']['link_state_dim'])])

            lens = tf.math.segment_sum(data=tf.ones_like(paths),
                                       segment_ids=paths)

            # Generate the aforementioned tensor [n_paths, max_len_path, dimension_link]
            link_inputs = tf.scatter_nd(ids, h_tild, shape)

            # Define the RNN used for the message passing links to paths
            gru_rnn = tf.keras.layers.RNN(self.path_update,
                                          return_sequences=True,
                                          return_state=True)

            # First message passing: update the path_state
            outputs, path_state = gru_rnn(inputs=link_inputs,
                                          initial_state=path_state,
                                          mask=tf.sequence_mask(lens))

            # For every link, gather and sum the sequence of hidden states of the paths that contain it
            m = tf.gather_nd(outputs, ids)
            m = tf.math.unsorted_segment_sum(m, links, f_['n_links'])

            # Second message passing: update the link_state
            link_state, _ = self.link_update(m, [link_state])

        # Call the readout ANN and return its predictions
        shape = tf.stack([
            f_['n_paths'],
            int(self.config['HYPERPARAMETERS']['path_state_dim']) - 4
        ], axis=0)



        congestion = self.readout(link_state, training=training)
        dly = self.readout(path_state, training=training)
        
        # print("SHAPE OF R: ",r.shape)
        # Initialize the initial hidden state for paths
        model_2 = tf.concat([
            tf.expand_dims(f_['arrival_time'], axis=1),
            tf.expand_dims(f_['duration'], axis=1),
            tf.expand_dims(f_['deadline'], axis=1),
            tf.expand_dims(f_['volume'], axis=1),
            tf.zeros(shape)
        ], axis=1)
        r = self.model_2(model_2,training=training)
        print("Congestion: ",congestion.shape)
        print("Delay: ",dly.shape)
        print("R: ",r.shape)

       
        # model_3 = tf.concat([
        #   tf.expand_dims(reshaped_tensor1,axis=-1),
        #   tf.expand_dims(reshaped_tensor2,axis=-1),
        #   # tf.expand_dims(dly,axis=-1),
        # ],axis=-1)
       # d = self.deadline_detection(model_2,training=training)
        # shape = tf.stack([
        #     f_['n_paths'],
        #     int(self.config['HYPERPARAMETERS']['path_state_dim']) - 2
        # ], axis=-1)
        # model_3 = tf.concat([
        #     tf.expand_dims(congestion, axis=2),
        #     tf.expand_dims(r, axis=2),
        #     tf.expand_dims(dly, axis=2),
        #     # tf.zeros(shape)
        # ], axis=2)
        # d = self.deadline_detection(model_3,training=training)
        return r

        # shape = tf.stack([
        #     f_['n_paths'],
        #     int(self.config['HYPERPARAMETERS']['path_state_dim']) - 4
        # ], axis=0)

        # # Initialize the initial hidden state for paths
        # model2 = tf.concat([
        #     tf.expand_dims(f_['arrival_time'], axis=1),
        #     tf.expand_dims(f_['duration'], axis=1),
        #     tf.expand_dims(f_['volume'], axis=1),
        #     tf.expand_dims(f_['deadline'], axis=1),
        #     tf.zeros(shape)], axis=1)

        # deadline_prediction = self.deadline_detection(model2,training=training)
        # return deadline_prediction


        # congestion = self.congestion_prediction(link_state, training=training)
        #print(r)
       # r = self.congestion_prediction(r)
        #return r
      #  print(len(r),"Delay########")
        # print(len(congestion),"Congestion########")
        # return r,congestion


def r_squared(labels, predictions):
    """Computes the R^2 score.

        Args:
            labels (tf.Tensor): True values
            labels (tf.Tensor): This is the second item returned from the input_fn passed to train, evaluate, and predict.
                                If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed.

        Returns:
            tf.Tensor: Mean R^2
        """

    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    r_sq = 1.0 - tf.truediv(unexplained_error, total_error)

    # Needed for tf2 compatibility.
    m_r_sq, update_rsq_op = tf.compat.v1.metrics.mean(r_sq)

    return m_r_sq, update_rsq_op


def model_fn(features, labels, mode, params):
    """model_fn used by the estimator, which, given inputs and a number of other parameters,
       returns the ops necessary to perform training, evaluation, or predictions.

    Args:
        features (dict): This is the first item returned from the input_fn passed to train, evaluate, and predict.
        labels (tf.Tensor): This is the second item returned from the input_fn passed to train, evaluate, and predict.
                            If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed.
        mode (tf.estimator.ModeKeys): Specifies if this is training, evaluation or prediction.
        params (dict): Dict of hyperparameters. Will receive what is passed to Estimator in params parameter.

    Returns:
        tf.estimator.EstimatorSpec: Ops and objects returned from a model_fn and passed to an Estimator.
    """

    # Create the model.
    model = Experimental_RouteNet_Model(params)

    # Execute the call function and obtain the predictions.
    predictions = model(features, training=(mode == tf.estimator.ModeKeys.TRAIN))

    #predictions = tf.squeeze(predictions)
    #print(predictions)
    
    # If we are performing predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predicted values.
        return tf.estimator.EstimatorSpec(
            mode, predictions={
                #'delay': predictions[0],
                'congestion':predictions#[1]
            })

    # Define the loss function.
    loss_function = tf.keras.losses.MeanSquaredError()

    # Obtain the regularization loss of the model.
    regularization_loss = sum(model.losses)

    # Compute the loss defined previously.
    loss = loss_function(labels, predictions)#[0])

    # Compute the total loss.
    total_loss = loss + regularization_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('total_loss', total_loss)

    # If we are performing evaluation.
    if mode == tf.estimator.ModeKeys.EVAL:
        # Define the different evaluation metrics
        label_mean = tf.keras.metrics.Mean()
        _ = label_mean.update_state(labels)
        prediction_mean = tf.keras.metrics.Mean()
        _ = prediction_mean.update_state(predictions)#[0])
        mae = tf.keras.metrics.MeanAbsoluteError()
        _ = mae.update_state(labels, predictions)#[0])
        mre = tf.keras.metrics.MeanRelativeError(normalizer=tf.abs(labels))
        _ = mre.update_state(labels, predictions)#[0])

        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={
                'label/mean': label_mean,
                'prediction/mean': prediction_mean,
                'mae': mae,
                'mre': mre,
                'r-squared': r_squared(labels, predictions)#[0])
            }
        )

    # If we are performing training.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Compute the gradients.
    grads = tf.gradients(total_loss, model.trainable_variables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in model.trainable_variables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    # Define an exponential decay schedule.
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(float(params['HYPERPARAMETERS']['learning_rate']),
                                                                int(params['HYPERPARAMETERS']['decay_steps']),
                                                                float(params['HYPERPARAMETERS']['decay_rate']),
                                                                staircase=True)

    # Define an Adam optimizer using the defined exponential decay.
    optimizer = tf.keras.optimizers.Adam(learning_rate=decayed_lr)
    
    # Manually assign tf.compat.v1.global_step variable to optimizer.iterations
    # to make tf.compat.v1.train.global_step increased correctly.
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    # Apply the processed gradients using the optimizer.
    train_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Define the logging hook. It returns the loss, the regularization loss and the
    # total loss every 10 iterations.
    logging_hook = tf.estimator.LoggingTensorHook(
        {"Loss": loss,
         "Regularization loss": regularization_loss,
         "Total loss": total_loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )

