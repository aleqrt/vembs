from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Define the MLP model
def create_mlp(input_dim, n_class):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    if n_class == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(n_class, activation='softmax'))
    return model
