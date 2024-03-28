import sklearn.linear_model as lm
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
import tensorflow as tf

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class MyModel:
    def __init__(self, input_shape):
        super().__init__()
        # Create the model layers
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, input_shape=input_shape),
            tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-07,
                                             amsgrad=False)

        self.model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

        # Model checkpoint callback to save the best model based on lowest MAE
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                             monitor='mae',
                                                             verbose=1,
                                                             save_best_only=True,
                                                             mode='min')

    def train(self, X_train, y_train, epochs=1000, batch_size=32):

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 callbacks=[self.checkpoint])
        return history

    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

    def predict(self, X_new):
        return self.model.predict(X_new)

    def load_best_model(self):
        # Loads the best model saved during training based on lowest validation MAE.
        self.model = tf.keras.models.load_model('best_model.h5')


def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# Data Summary Function
def Data_Summary(data):
    print("Data Summary")
    print("=====================================")
    print("First 5 rows")
    print(data.describe())
    print("=====================================")
    print("Data types")
    print(data.info())
    print("=====================================")
    print("Data count")
    print(data.count())
    print("=====================================")
    print("Missing values")
    print(data.isnull().sum())
    print("=====================================")
    print("Data shape")
    print(data.shape)
    print("=====================================")
    print("Unique values in each column")
    print(data.nunique())
    print("=====================================")

    # describe the last column
    print("Last column description")
    print(data.iloc[:, -1].describe())
    # Visualize the last column
    temp_data = data.drop(data.index[0])
    plt.figure(figsize=(7, 6))
    plt.bar(temp_data.iloc[:, -1].unique(), temp_data.iloc[:, -1].value_counts())
    plt.show(block=True)

    print("---------------------------------------------------------------------------------")


def Machine_Learning(X_train, y_train, X_test, y_test):

    log_reg = lm.LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_score = log_reg.score(X_test, y_test)
    print("Logistic Regression Testing Accuracy: ", log_reg_score)

    lin_reg = lm.LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_score = lin_reg.score(X_test, y_test)
    print("Linear Regression Testing Accuracy: ", lin_reg_score)

    Sgd_reg = lm.SGDRegressor()
    Sgd_reg.fit(X_train, y_train)
    Sgd_reg_score = Sgd_reg.score(X_test, y_test)
    print("SGD Testing Accuracy: ", Sgd_reg_score)

    Ridge_reg = lm.Ridge()
    Ridge_reg.fit(X_train, y_train)
    Ridge_reg_score = Ridge_reg.score(X_test, y_test)
    print("Ridge Testing Accuracy: ", Ridge_reg_score)

    Lasso_reg = lm.Lasso()
    Lasso_reg.fit(X_train, y_train)
    Lasso_reg_score = Lasso_reg.score(X_test, y_test)
    print("Lasso Testing Accuracy: ", Lasso_reg_score)

    Elastic_reg = lm.ElasticNet()
    Elastic_reg.fit(X_train, y_train)
    Elastic_reg_score = Elastic_reg.score(X_test, y_test)
    print("Elastic Testing Accuracy: ", Elastic_reg_score)

    Huber_reg = lm.HuberRegressor()
    Huber_reg.fit(X_train, y_train)
    Huber_reg_score = Huber_reg.score(X_test, y_test)
    print("Huber Testing Accuracy: ", Huber_reg_score)

    Ransac_reg = lm.RANSACRegressor()
    Ransac_reg.fit(X_train, y_train)
    Ransac_reg_score = Ransac_reg.score(X_test, y_test)
    print("Ransac Testing Accuracy: ", Ransac_reg_score)

    Theil_reg = lm.TheilSenRegressor()
    Theil_reg.fit(X_train, y_train)
    Theil_reg_score = Theil_reg.score(X_test, y_test)
    print("Theil Testing Accuracy: ", Theil_reg_score)

    models = [log_reg, lin_reg, Sgd_reg, Ridge_reg, Lasso_reg, Elastic_reg, Huber_reg, Ransac_reg, Theil_reg]
    scores = [log_reg_score, lin_reg_score, Sgd_reg_score, Ridge_reg_score, Lasso_reg_score, Elastic_reg_score, Huber_reg_score, Ransac_reg_score, Theil_reg_score]

    return models, scores


def Preprocessing(data):
    print("Preprocessing")
    print("=====================================")
    std_scaler = StandardScaler()

    # # change first column type to str
    # data.iloc[:, 0] = data.iloc[:, 0].astype(str)
#
    # # change values in the first column to categorise
    # data.iloc[:, 0] = data.iloc[:, 0].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape: ", y_train.shape)
    print("y_test shape: ", y_test.shape)

    print(X_train.head())
    print(X_test.head())
    print(y_train.head())
    print(y_test.head())
    print("---------------------------------------------------------------------------------")

    return X_train, X_test, y_train, y_test


# In this function deletes the rows that have the repeated V1 and v2 values compared to the last row
def Delete_rows(data):
    print("Deleting rows")
    print("=====================================")
    data = data.drop_duplicates(subset=['V1', 'V2'], keep='last')
    print(data.head())
    print("---------------------------------------------------------------------------------")
    return data

# in this function replaces the values in V1 with categorical values
def Process_Data(data):
    print("Processing Data")
    print("=====================================")

    data['V1'] = data['V1'].astype('category')
    data['V1'] = data['V1'].cat.codes

    data['V2'] = data['V2'].astype('category')
    data['V2'] = data['V2'].cat.codes

    print(data.head())
    print("---------------------------------------------------------------------------------")
    return data

# Create a dataframe with only 0's in V1 and V2 starts from 0 to 250, no V3
def Create_Data():
    print("Creating Data")
    print("=====================================")

    data = pd.DataFrame(columns=['V1', 'V2'])

    for i in range(0, 251):
        data = data._append({'V1': 0, 'V2': i}, ignore_index=True)

    print(data.head())
    print("---------------------------------------------------------------------------------")
    return data


if __name__ == '__main__':
    IPD = read_csv("Data/IPD.csv")
    # progression = read_csv("Data/progression.csv")
    IPD = Delete_rows(IPD)
    IPD = Process_Data(IPD)

    # saveIPD as a csv
    IPD.to_csv("Data/IPDtest.csv", index=False)

    Test_data = Create_Data()
    # print("-------------------------------------------------------------------------------------------------")
    # Data_Summary(progression)

    # print("IPD: ", IPD.head())

    X_train, X_test, y_train, y_test = Preprocessing(IPD)
#
    models, scores = Machine_Learning(X_train, y_train, X_test, y_test)

    print("Models: ", models)
    print("Scores: ", scores)

    Test_data_y = models[0].predict(Test_data)
    print("Test Data: ", Test_data)
    print("Test Data Predictions: ", Test_data_y)

    # save them into a csv in first column months from 0 to 250 and in the second column the predictions
    Test_data = pd.DataFrame(Test_data)
    Test_data = Test_data.rename(columns={0: 'V1', 1: 'ID'})
    Test_data.drop(columns=['V1'], inplace=True)
    Test_data['Predictions'] = Test_data_y
    Test_data.to_csv("Data/Test_data.csv", index=False)
