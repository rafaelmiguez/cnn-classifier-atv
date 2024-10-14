# Obtendo Dados

Sites como [Kaggle](https://www.kaggle.com/) e [images.cv](https://images.cv/) podem ser usadas para obter datasets de imagens.

# Realizando Pré-processamento

Pode ser necessário realizar um pré-processamento das imagens antes de usá-las para treinamento. Considere aplicar:

* Dimensionamento
* Mudança de escala de cor
* Normalização de matriz de imagens
* Data augmentation

# Obtendo Modelo

A biblioteca `keras.applications` possui uma série de modelos de deep learning listados [aqui](https://keras.io/api/applications/). É possível importar um modelo e usá-lo. Os pesos são baixados automaticamente.

## Fine Tunning

É possível ainda realizar um *fine tunning* no modelo escolhido para personalizá-lo ao seu problema. Para isso:

* Congele as camadas iniciais de forma que apenas as últimas camadas do modelo serão treinadas
* Adicione camadas do novo classificador
* Compile e treine o modelo

# Links Úteis

* [Unlock the Power of Fine-Tuning Pre-Trained Models in TensorFlow &amp; Keras](https://learnopencv.com/fine-tuning-pre-trained-models-tensorflow-keras)
* [Fine-tune InceptionV3 on a new set of classes](https://keras.io/api/applications/#finetune-inceptionv3-on-a-new-set-of-classes)
