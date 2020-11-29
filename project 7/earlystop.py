import tensorflow.keras as keras
import numpy as np
import math

class EarlyStopByF1(keras.callbacks.Callback):
    def __init__(self, snp_df, features, grouped_df, value = 0, verbose = 0, days_to_predict = 0, noof_missing_days = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.value = value
        self.verbose = verbose
        self.days_to_predict = days_to_predict
        self.noof_missing_days = noof_missing_days
        self.snp_df = snp_df
        self.features = features
        self.grouped_df = grouped_df



    def on_epoch_end(self, epoch, logs={}):
        noof_correct_movement = 0
        noof_predictions = 0
        diffs = []
        input_features = []

        for i in range(int(self.days_to_predict), int(self.noof_missing_days), -1):
            input_features.clear()
            noof_predictions += 1
            actual_change = self.snp_df.iloc[(-1 * i) + 1]['percent change']
            for feature in self.features:
                input_features.append(self.grouped_df.iloc[-1 * i][feature])
            predicted_change = self.model.predict(np.array([input_features]))[0, 0]

            if (predicted_change * actual_change) > 0:
                diffs.append(math.fabs(predicted_change - actual_change))
                noof_correct_movement += 1

        percent_correct = (noof_correct_movement / noof_predictions)


         #predict = np.asarray(self.model.predict(self.test_data[0]))
         #target = self.test_data[1]
         #score = f1_score(target, prediction)
        score = percent_correct
        if score > self.value:
            if self.verbose >0:
                print("Epoch %05d: early stopping Threshold" % epoch)
                print("Score is: {}", score)
                print("Default value is: {}", self.value)
            self.model.stop_training = True

