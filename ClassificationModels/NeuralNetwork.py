from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow import keras


def buildNNModel(data):
    X = data[['Parenting', 'Self-realization']]
    y = data['Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['Parenting', 'Self-realization']
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(preprocessor.fit_transform(X_train), y_train, epochs=50, batch_size=16, verbose=2)

    test_loss, test_accuracy = model.evaluate(preprocessor.transform(X_test), y_test, verbose=0)
    print(f'Accuracy:', test_accuracy)
