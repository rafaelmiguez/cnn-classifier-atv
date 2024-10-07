# Sobre o Projeto

Este é projeto de um pacote Python capaz de treinar um classificador CNN(Convolutional Neural Network) de imagens.

# Obtenção de Imagens

Sites como o [Kaggle](https://www.kaggle.com/) e [images.cv](https://images.cv/) podem ser usados para obter a base de dados das imagens. Note que pode ser interessante remover ou filtrar algumas imagens obtidas destes sites por não serem ideais para certas aplicações de ML.

# Configuração de Parâmetros

É possível configurar os parâmetros de pré-processamento e treino pelo arquivo `params.py`. Segue abaixo um exemplo desse arquivo.

```python
color_mode = 'RGB'
zoom_factor = 1.2
img_size = 64

optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
epochs = 10
```

* `color_mode`: Modo de cor das imagens pré-processadas. Pode ser 'L' para escala de cinza e 'RGB' para a escala RGB
* `zoom_factor`: Fator de zoom, usado para aplicar efeito de zoom a imagens no pré-processamento. Deixe como 1 caso não deseje aplicar este efeito
* `img_size`: Valor para o qual as imagens serão redimensionadas
* `optimizer`: Algoritmo usado para ajustar pesos do modelo durante treinamento. Ver mais [aqui](https://keras.io/api/optimizers/)
* `loss`: Função de perda (ou função custo) que mede o quanto as previsões do modelo estão se afastando dos valores esperados. Ver mais [aqui](https://keras.io/api/losses/)
* `epochs`: Quantidade de épocas do treinamento

As imagens de treinamento devem ser salvas na pasta `imgs`. Esta pasta deve conter subpastas que por sua vez conterão as imagens. O nome de cada subpasta irá representar o nome da classe das imagens contidas nela. Por padrão, a pasta `imgs` contem as subpastas `daisy`, `rose` e `sunflower`, contendo imagens de flores das respectivas classes.

A pasta `predict_imgs` deve conter imagens definidas pelo usuário que serão usadas para testar o modelo pronto. Por padrão, esta pasta contém imagens de flores dos tipos `daisy`(margaridas), `rose`(rosas) e `sunflower`(girassóis).

# Executando o Código

## Instalando Depedências

`poetry install`

## Realizando Pré-processamento

`poetry run preprocess`

Este comando irá gerar a pasta `tmp` contendo as imagens pré-processadas.

## Realizando Treinamento

`poetry run train`

Este comando irá exportar o modelo treinado para o arquivo `tmp/model.keras`.

## Classificando Imagens

`poetry run classify`

Este comando irá classificar as imagens geradas na pasta `predict_imgs`, printando os resultados a exemplo do visto abaixo.

```
daisy1.png: daisy
daisy2.png: daisy
daisy3.png: sunflower
daisy4.png: rose
daisy5.png: daisy
daisy6.png: daisy
rose1.png: rose
rose2.png: rose
rose3.png: rose
rose4.png: rose
rose5.png: rose
rose6.png: rose
sunflower1.png: sunflower
sunflower2.png: sunflower
sunflower3.png: rose
sunflower4.png: sunflower
sunflower5.png: daisy
sunflower6.png: sunflower
```
