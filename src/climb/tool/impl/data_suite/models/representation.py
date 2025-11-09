# import keras
# from keras import layers

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


class AutoEncoder:
    def __init__(self, input_shape, encode_dim):
        self.input_shape = input_shape
        self.encode_dim = encode_dim
        self.autoencoder, self.encoder = self._build_model()
        self.autoencoder.compile(optimizer="adam", loss="mse")

    def _build_model(self):
        """
        We create a model with an input layer, a middle layer, and an output layer. The middle layer is
        the encoded bottleneck representation of the input. The output layer is the decoded representation of the
        input

        Returns:
          The autoencoder and the encoder.
        """

        encoding_dim = self.encode_dim
        input_data = keras.Input(shape=(self.input_shape,))

        middle = layers.Dense(encoding_dim, activation="relu")(input_data)
        middlex = layers.Dense(
            int(encoding_dim) * 2,
            activation="relu",
        )(middle)
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(encoding_dim, activation="relu")(middlex)

        middle2 = layers.Dense(encoding_dim, activation="relu")(encoded)

        middley = layers.Dense(
            int(encoding_dim) * 2,
            activation="relu",
        )(middle2)

        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(self.input_shape, activation="sigmoid")(middley)

        autoencoder = keras.Model(input_data, decoded)

        encoder = keras.Model(input_data, encoded)

        return autoencoder, encoder

    def fit(self, x_train):
        """
        The function takes in the training data and trains the autoencoder for 100 epochs with a batch
        size of 8

        Args:
          x_train: The training data
        """
        self.autoencoder.fit(
            x_train,
            x_train,
            epochs=100,
            batch_size=8,
            shuffle=True,
        )

    def bottleneck(self, x_test):
        """
        The bottleneck function takes an input and returns the bottleneck compressed representation

        Args:
          x_test: The input data to be encoded.

        Returns:
          The bottleneck features of the input data.
        """
        return self.encoder.predict(x_test)


def compute_representation(
    train,
    test,
    copula_samples,
    n_components=2,
    rep_type="pca",
    seed=42,
):
    """
    > This function takes in the training and test data, the copula samples, and the number of components to use
    for the representation. It then standardizes the data, and uses either PCA or an autoencoder to
    compute the representation

    Args:
      train: the training data
      test: the test data
      copula_samples: the samples from the copula
      n_components: the number of dimensions to reduce to. Defaults to 2
      rep_type: the type of representation to use. Can be either "pca" or "ae". Defaults to pca
      seed: random seed. Defaults to 42

    Returns:
      the train, test and copula samples in the new representation.
    """

    scaler = StandardScaler()
    scaler.fit(train)

    combined_X_train_sc = scaler.transform(train)

    combined_X_test_sc = scaler.transform(test)

    copula_sc = scaler.transform(copula_samples)

    if rep_type == "pca":
        pca = PCA(n_components=n_components, random_state=seed)
        pcs_train = pca.fit_transform(combined_X_train_sc)
        pcs_test = pca.transform(combined_X_test_sc)
        pcs_copula = pca.transform(copula_sc)

    if rep_type == "ae":
        ae = AutoEncoder(
            input_shape=combined_X_train_sc.shape[1],
            encode_dim=n_components,
        )
        ae.fit(combined_X_train_sc)
        pcs_train = ae.bottleneck(combined_X_train_sc)
        pcs_test = ae.bottleneck(combined_X_test_sc)
        pcs_copula = ae.bottleneck(copula_sc)

    return pcs_train, pcs_test, pcs_copula


def representation_class_based(
    train,
    copula_samples,
    n_components=2,
    rep_type="pca",
    seed=42,
):
    """
    > This function computes a representation of the data. It first standardize the training data and the copula samples, then we apply PCA to the
    standardized data, and finally we return the PCA components of the training data, the PCA components
    of the copula samples, the PCA object, and the scaler object

    Args:
      train: the training data
      copula_samples: the samples from the copula
      n_components: The number of components to keep. Defaults to 2
      rep_type: the type of representation to use. Currently only PCA is supported. Defaults to pca
      seed: the random seed. Defaults to 42

    Returns:
      the transformed training data, the transformed copula samples, the PCA object, and the scaler
    object.
    """

    scaler = StandardScaler()
    scaler.fit(train)

    combined_X_train_sc = scaler.transform(train)

    copula_sc = scaler.transform(copula_samples)

    if rep_type == "pca":
        pca = PCA(n_components=n_components, random_state=seed)
        pcs_train = pca.fit_transform(combined_X_train_sc)
        pcs_copula = pca.transform(copula_sc)

    return pcs_train, pcs_copula, pca, scaler
