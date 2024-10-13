# Obtendo Dados

Sites como [Kaggle](https://www.kaggle.com/) e [images.cv](https://images.cv/) podem ser usadas para obter datasets de imagens.

# Realizando Pré-processamento

Pode ser necessário realizar um pré-processamento das imagens antes de usá-las para treinamento. Considere aplicar:

* Dimensionamento
* Mudança de escala de cor
* Normalização de matriz de imagens
* Data augmentation

# Treinamento e Testando Modelo

* Codifique os `labels` das imagens caso não sejam numéricos. A função `LabelEncoder` da biblioteca `sklearn.preprocessing` pode ser usada para isso
* Separe os dados em treino e teste. A função `train_test_split` da biblioteca `sklearn.model_selection` pode ser usada
* Crie um modelo sequencial usando a classe `Sequential` da biblioteca `keras.models` e adicione as camadas de convolução usando a api `layers` da biblioteca `keras`
  * É interessante inserir camadas de *data augmentation* como `layers.RandomFlip`, `layers.RandomRotation` e `layers.RandomZoom`, além de uma cadamada de *drop out* `layers.dropout` antes das camadas densas para evitar *overfitting.*
* Compile o modelo passando os parâmetros desejados de `optimizer`, `loss` e `accuracy` usando o método `Sequential.compile`
* Treine o modelo com os dados de treinamento definindo a quantiade de épocas usando o método `Sequential.fit`
* Avalie o modelo usando o método `Sequential.evaluate`
* Se o modelo for satisfatório, exporte-o o modelo usando o método `Sequential.save`
* Teste o modelo usando o método `Sequential.predict`

# Links Úteis

* [TensorFlow - Rede Neural Convolucional (CNN)](https://www.tensorflow.org/tutorials/images/cnn?hl=pt-br)
* [Creating a CNN Model for Image Classification with TensorFlow](https://medium.com/@esrasoylu/creating-a-cnn-model-for-image-classification-with-tensorflow-49b84be8c12a)
* [TensorFlow - Classificação de imagem](https://www.tensorflow.org/tutorials/images/classification?hl=pt-br)
