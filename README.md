
# Restauration d'images par Deep Learning



## 1. Présentation des données

<p> Nous avons à notre disposition 5500 photos dégradées. Pour chacunes de ces photos, une version restaurée est également disponible. </p>
<img src="https://imgur.com/anIvuZ3.png"> </img>

<p>Image dégradée</p>
<img src="https://imgur.com/X4XZ5ID.png"> </img>
<p>Image restaurée</p>
<p> Un autre jeu de données nous a également été fourni, celui-ci pour tester le modèle. Il s'agit de 4000 images dégradées. La version non dégradée de ces images n'est pas disponible.</p>
<p> Les images ont toutes la même dimension de 576x720 pixels, et elles sont en couleur (RGB). Le nombre de feature par image est donc de 576 * 720 * 3 = 1 244 160 valeurs.</p>



## 2. Définition du problème

<p> Comme nous avons à notre disposition pour chacune des images d'entrainement une version restaurée et degradée, nous sommes en mesure de définir le problème comme étant un problème supervisé. </p>
<p> Comme features X nous auront l'ensemble des valeurs composant une image dégradée, et le label Y sera composé de l'ensemble des valeurs composant la version restaurée de cette image </p>
<p> Nous pourrons ensuite entrainer un modèle afin d'obtenir la version restaurée d'une image à partir de sa version dégradée </p>
<p> Ce type de problème est un problème de regression. Le nombre de classe possible pour une feature est égale au nombre de feature, il est donc impossible de faire de la classification catégorique. Notre modèle doit avoir en sortie une image restaurée, donc un array de  244 160 valeurs numériques. </p>



## 3. Préparation des données

<p> Comme nous travaillons avec des images, il n'a pas été possible de chercher des corrélations entre nos features car ce sont uniquement des valeurs de pixels. </p>
<p> Il est cependant fortement conseillé de normaliser les valeurs des pixels afin d'aider notre modèle à converger vers une solution optimale. Nous avons donc à diviser l'ensemble des pixels par leur valeur maximale, à savoir 255. De cette manière nous obtenons une valeur entre 0 et 1 pour chacun des pixels.</p>
<p> Comme nous travaillons avec un important volume de données (5500 * 2 * 1 244 160 = 13 685 760 000 valeurs à charger en mémoire), nous utiliserons un générateur de données. </p>
<p> Un générateur de données permet de charger un batch de données d'une taille fixe depuis notre système de fichier. Ce batch est ensuite passé à notre modèle afin qu'il puisse s'entrainer. Une fois la backpropagation terminée, un autre batch est chargé en mémoire et est passé au modèle, et ce jusqu'à ce que l'ensemble des données d'entrainement soit parcouru. </p>
<p> La librairie Keras offre une classe Data Generator permettant de faire cette manipulation. Cependant, cette classe est pensée pour des problèmes de classification catégorique, elle n'est donc pas utilisable pour notre type de problème. </p>
<p> Nous avons donc développé notre propre générateur de données, dont la définition est présente ci-dessous.</p>



```python
#appelé à chaque fois qu'on charge une image en mémoire
def load_img(path):
    #récupère l'ensemble des pixels de l'image et les stockent dans une variable Python
    img = imread(path)
    #normalise les valeurs des pixels
    img = img/ 255
    return img

def img_generator(img_list,batch_size):
    while True:
        X_batch_input = []
        y_batch_input = []
        degraded_path =  '/home/fablab/dataset_clean_degraded/degraded/'
        clean_path =  '/home/fablab/dataset_clean_degraded/clean/'
        #choisi aléatoirement des noms d'images égale à la taille de notre batch
        batch_path = np.random.choice(a = img_list, size= batch_size)
        #pour chacune des images de notre batch
        for img_name in batch_path:
            #charge l'image dégradée en mémoire
            current_X = load_img(degraded_path+img_name)
            #charge l'image clean correspondante en mémoire
            current_y = load_img(clean_path+img_name)
            X_batch_input += [ current_X ]
            y_batch_input += [ current_y ]
        batch_X = np.array(X_batch_input)
        batch_y = np.array(y_batch_input)
        #renvoie un batch
        yield (batch_X, batch_y)

```

 Nous avons choisi 4 en batch size car c'est un multiple de 5500. De ce fait, le nombre de step par epoch est un nombre entier ( 5500 / 4 = 1375 ).



Pour chaque modèle, il faut : 
- La définition du modèle ( voir : https://keras.io/visualization/ )
- Les paramètres d'entrainement 
    Hyperparameter Tweaking : Filter size, Padding, etc.. 
    Nb epochs, batch size, step.
- Analyses des performances
    Métriques d'évaluations ( mse, binary crossentropy, etc..)
    Learning curve (voir si l'entrainement "cap" au fil des epochs)
    Plot predictions
- Pistes d'améliorations
    Modifier les layers du modèle, etc...

#4. Choix du modèle

##4.1 Convolutional Neural Network


###4.1.1 Pourquoi utiliser un CNN ?
Notre problème s'inscrit dans le domaine de la Computer Vision. Ce domaine utilise majoritairement des modèles de réseaux neuronaux à convolution. Nous nous sommes donc orientés sur ce type de modèle en premier lieu. 


###4.1.2 Comment fonctionne un CNN ?

<br>
<img src="https://imgur.com/dszWMJt.png">
<br>

La couche de convolution permet d'extraire les high feature d'un image. Pour ce faire elle va parcourir l'input (ici une image) avec un filtre (ou kernel).

Ce filtre est en réalité une matrice d'une taille que nous avons à définir au préalable (attribut kernel_size avec Keras). A chaque déplacement du filtre sur l'image, une multiplication matricielle entre le filtre et sa position dans l'image est effectuée. La somme de l'ensemble des valeurs composant ce produit matriciel  est alors enregistrée dans une nouvelle matrice. Une fois l'image parcourue, cette matrice est renvoyée, il s'agit de l'image convoluée. 

Le déplacement du filtre sur l'image se fait de haut en base et de gauche à droite, et le nombre de pixels parcouru après un déplacement est paramétrable via l'attribut stride dans Keras. 

Cette opération réduit forcément la dimension de l'image, plus le filtre est grand plus on perd en dimension. Un moyen de contrer ça est d'ajouter des valeurs égales à zero autour de notre image. De ce fait, le filtre parcourera suffisament l'image pour créer une image convoluée de la même taille que l'image de base. Ce paramètre est modifiable dans Keras via l'attribut padding, il peux être égal à same (dans le cas ou l'on veux un output de taille égale à l'input) ou valide (l'image est alors réduite). 

Dans le cas où notre image a plusieurs channels (ici 3 pour les pixels RGB), nous aurons 3 filtres identiques qui parcourront l'image de manière identique, mais chacun sera sur un cannal (ici une couleur) qui lui est propre. Le résultat de l'opération matricielle de chacun de ces filtres est additionné puis enregistré dans une nouvelle matrice de la même manière que précédemment. 

La couche de convolution est souvent suivie par une couche appelée "Pooling Layer". Cette couche permet de réduire les dimensions de l'image convoluée. Elle parcourt l'image avec un filtre et extraie une valeur à la fois dans une nouvelle matrice. Cette matrice sera renvoyée par la couche et sera de taille plus ou moins réduite en fonction de la taille du filtre (paramétrable dans Keras). 
La valeur renvoyée à chaque déplacement du filtre peut être soit la valeur maximale observée (MaxPoolingLayer) soit la moyenne des valeurs observée(AveragePoolingLayer).

 Cette couche permet deux choses : 
 - Réduire le temps d'entrainement car il y a moins de variables à paramétrer.
 - Faire ressortir les features dominantes de l'image.
Dans le cas du MaxPooling, il fait aussi l'effet d'un reducteur de bruit. Nous l'utiliserons donc car nous cherchons à réduire le bruit sur nos images.

Ces deux couches sont généralement suivies par un DenseLayer (ou Fully Connected Layer). Celui-ci va permettre de mapper les features extraites avec un nombre défini de classe. Il est alors possible de résoudre des problèmes de classification. Nous ne pouvons pas utiliser cette architecture car nous devons obtenir une image de la même dimension que notre entrée. 

Cependant, nous pouvons utiliser ce type d'architecture afin de classifier nos images comme étant "clean" ou "dégradée". Il sera alors peut-être possible d'extraire les features que le modèle utilise pour entrainer un autre modèle. 


###4.1.3 Comment l'avons-nous implémenté ?

Notre classifieur binaire est constitué de 4 couches de convolution ainsi que de 2 couches de neurones entièrement connectées.

<img src="https://i.imgur.com/ghMfQ7F.png">


Ci-dessous, vous trouverez le code utilisé pour implémenter ce réseau de neurones avec la bibliothèque Keras :
```python
# Initialisation
input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
classifier = Sequential()


# Couches de convolution

## Couche 1
### Convolution :
#### filters = dimension de l'espace de sortie : (32)
#### kernel_size = paramètres de la fenêtre de convolution : (3, 3)
classifier.add(Conv2D(32, (3, 3), input_shape=(IMG_HEIGHT,IMG_WIDTH,3) , activation = 'relu'))

### Pooling :
#### pool_size = facteurs de reduction d'echelle (vertical, horizontal) : (2, 2)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

## Couche 2
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

## Couche 3
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

## Couche 4
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Couches entièrement connectées

## Applatissage (passage en 1D)
classifier.add(Flatten())

## Dense layers
### units = dimension de l'espace de sortie : (64)
classifier.add(Dense(units = 64, activation = 'relu', name ='feature_dense'))
### units = dimension de l'espace de sortie : (1)
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# Compilation
classifier.summary()
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```


Voici le résumé du modèle :

<img src="https://imgur.com/WmspSEm.png">


Pour entrainer le modède, on execute le code suivant :

```python
#Model training
classifier.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = EPOCHS)
```

Nous obtenons une précision de 0.9920 avec le paramètre BATCH_SIZE défini à 16 et après 10 époques d'entrainement.


<img src="https://imgur.com/gUE1w0X.png">


Pour extraire les features de notre modèle, on récupère ce dernier en prenant soin de s'assurer que la sortie correspondra à l'avant dernière couche, soit la couche contenant les features à extraire.

```python
from keras.models import Model,load_model

#Préparation du modèle intermédiaire
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('feature_dense').output)
intermediate_layer_model.summary()
```

Puis on execute le code suivant afin d'obtenir les features sous forme d'un tableau de données.

```python
#Obtenir les features

feature_engg_data = intermediate_layer_model.predict(train_generator)
feature_engg_data = pd.DataFrame(feature_engg_data)
print('feature_engg_data shape:', feature_engg_data.shape)
feature_engg_data.head(5)
```

### 4.2 Autoencoder

Le fait d'avoir à renvoyer une image et non une valeur catégorique nous a orienté vers les modèles d'auto encoders.

Un autoencodeur est, par définition, une technique pour encoder quelque chose automatiquement. En utilisant un réseau de neurones, l'autoencodeur est capable d'apprendre à décomposer des données (dans notre cas, des images) en assez petits morceaux de données, puis en utilisant cette représentation, pour reconstruire les données originales aussi près que possible de l'originale.

Cette tâche comporte deux éléments clés:
<ul>
  <li> 
  Encodeur: apprend à compresser l'entrée d'origine en un petit encodage 
  </li>
  <li>
  Décodeur: apprend à restaurer les données d'origine à partir de l'encodage généré par l'encodeur.
  </li> 
</ul>

Ces deux éléments sont entrainés ensemble pour obtenir le resultat le plus efficace des images dégradées à partir desquelles nous pouvons reconstruire des images de meilleure qualité.

Ci dessous le schéma de notre modèle implémentée sur Keras. La couche UpSampling2D est celle qui va rétablir les dimensions de notre image. 
<img src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1522830223/AutoEncoder_kfqad1.png" style = "display: block,margin-left:auto,margin-right: auto,width: 50%"></img>

Notre métrique d'évaluation du modèle est la Mean Squared Error, car elle permet de calculer la différence entre deux matrices numériques.



### Model Plot

<img src="https://i.imgur.com/SjIDCne.png" style = "display: block,margin-left:auto,margin-right: auto,width: 50%, height: 50%"> </img>



### 4.3 ESR CNN

Ce modèle est utilisé pour augmenter la résolution d'une image. Nous l'avons utilisé sur notre problème car nous avions du flou sur nos prédictions lorsque nous utilisions un encoder classique. 

Voici la définition du modèle :

### Model plot
<img src="https://i.imgur.com/6GzEwFk.png" height = 500>

####La learning curve :<br/>
<img src="https://imgur.com/qd9Ai7H.png"><br/>

####Les prédictions sur le jeu de donnée d'entrainement :
<br/>
<img src="https://imgur.com/3W996yk.png"><br/>

####ET sur le jeu de donnée de test :
<br/>
<img src="https://imgur.com/ApbfAqV.png"><br/>


### 4.4 Deep Denoiser SR CNN
Ce modèle a plus de couche qu'un autoencoder classique. Nous avons choisi de l'utiliser après nous être rendu compte que nous devions surement augmenter la profondeur de notre modèle après avoir utilisé un autoencoder classique.  Voici le schéma du modèle :
### Model plot
<img src="https://imgur.com/aqwe37y.png" height = 1200>
<br/>
Voici sa learning curve :<br/>
<img src="https://imgur.com/GxlTgGP.png"><br/>
On s'apercois que le modèle converge à :<br/>

<br/>
Voici des prédictions faites sur le jeu de donéne d'entrainement :<br/> 
<img src="https://imgur.com/2HMT9tj.png">
<br/>
Et sur le jeu de test : <br/>
<img src="https://imgur.com/stdg7vB.png">
<br/>
On s'aperçois que ce modèle est performant dans l'harmonisation des couleurs. 



### 4.5 VGG16
**VGG** est un modèle de classification  qui peut être chargé depuis la librairie deep learning Keras.<br/>
En utlisant cette interface, nous avons récupéré le modèle VGG et les couches pré-entrainées. Pour y ajouter notre ESR CNN, nous avons dû rétablir la sortie du modèle pour pouvoir mettre en entrée des images de notre format plus grand car VGG utilise des images de taille 224 x 224.<br/>



```python
#MODEL DEFINITION
model = Sequential()
init = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
#Récuperation de VGG16 selon notre type d'input
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
init_1 = layer_dict['block3_pool'].output
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(init_1)
x = UpSampling2D((2, 2))(x)

# DeConv2
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

# Deconv3
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
level1_1 = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
level2_1 = Convolution2D(16, (3, 3), activation='relu', padding='same')(level1_1)
level2_2 = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(level2_1)
level2 = Add()([level2_1, level2_2])
level1_2 = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(level2)
level1 = Add()([level1_1, level1_2])
decoded = Convolution2D(3, (5, 5), activation='linear', padding='same')(level1)
model = Model(base_model.input, decoded)
```



<img src="https://i.imgur.com/KxBAdPk.png" height=1500/>
<br/>
Le résultat est moins performant que notre CNN sans pré-entrainement car VGG est un modèle de classification donc les couches sont entrainées à faire de la classification or nous avons un probleme de regression.
<br/>
Learning Curve :<br/>
<img src="https://imgur.com/UUKdJXn.png"/>



```python

```

# Les modèles qui se sont avérés ne pas fonctionner :


### 4.6 LSTM
L'idée a ensuite été proposée d'utiliser une architecture en Long Short-Term memory.

La structure a été pensée comme suivant:


<img src="https://i.imgur.com/GZxonf9.png">
<br><br>


La loss function selectionnée etait la mean squared loss, exprimée par:
<br>
<img src="https://i.imgur.com/NXDkf7a.png">
<br>

La fonction d'activation quant à elle, était la fonction reLu.<br>
Une couche de convolution, suivit d'un max pooling et d'une deuxième convolution et  d'un deuxième max pooling. le tout redirigé sur une couche de Hidden layer fully connected redirigée sur différentes couches LSTM.

<br>
<img src="https://i.imgur.com/UYkW97l.png">
<br><br>

Le problème est apparu au moment de regrouper les deux réseaux, les couche LSTM fonctionnant en timedistributed, il a fallu transformer la sortie du CNN encodeur en séquences. La dernière action est impossible, les différentes images du dataset n'ayant aucun lien entre elles, le LSTM ne pouvait faire émerger aucun résultat valide et exploitable.


### 4.7 GAN

### 4.7.1 Définition
Generative Adversarial Networks (GANs) sont des architectures qui utilisent deux réseaux de neurones l'un contre l'autre dans l'objectif de générer de nouvelles instances synthétiques d'éléments qui passent pour de réelles données.
Cette architecture peut mimer tout type de distribution de la donnée, entrainant la possibilité de créer des éléments proches du monde réel comme des images, vidéos, sons etc.
Cette technique est à la racine du fonctionnement de DeepFake.

<img src="https://pathmind.com/images/wiki/gan_schema.png"></img>

### 4.7.2 Model plot
<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/04/Plot-of-the-Discriminator-Model-in-the-GAN.png"></img>

Exemple d'un GAN à une dimension

###4.7.3 Explication de l'échec

Le modèle du GAN s'est avéré ne pas être adapté au cadre de la restoration d'images. En effet, cette architecture est spécialisée dans la génération d'images ou de sons, en créant des éléments dans l'output qui n'étaient pas présent dans l'input. Il est du propre des GANs de générer des scènes peu réalistes avec une très large partie à la création interprétative.
Dans le cadre d'une restoration d'images, une modification de ce type n'est donc pas à favoriser.

# Pickle: Sérialisation d'objets en Python



Le module [pickle](https://docs.python.org/2/library/pickle.html) met en œuvre des protocoles binaires pour sérialiser et désérialiser une structure d'objet Python. 

Le **pickling** est le processus par lequel une hiérarchie d'objets Python est convertie en un flux d'octets, et le **unpickling** est l'opération inverse, par laquelle un flux d'octets (provenant d'un fichier binaire ou d'un objet de type octet) est reconverti en une hiérarchie d'objets. 

Le module Pickle offre les fonctions suivantes pour rendre le processus de Pickling plus pratique :

```python
pickle.dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None)
```
`pickle.dump()` permet d'écrire la représentation "pickled" de l'objet `obj` dans le fichier objet du fichier ouvert.

```python
 pickle.load(file, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None)
```
`pickle.load()` permet de le lire la représentation "pickled" d'un objet dans le fichier d'objets ouvert et renvoie la hiérarchie d'objets reconstituée qui y est spécifiée.



## Pickling

Pour pickle le modèle de notre choix, il suffira donc d'exécuter les lignes suivantes:

```python
import pickle
pickle.dump( model, open( "save-pickle-DDSRCNN-20200122.p", "wb"))
```




## Unpickling

Pour unpickle le modèle de notre choix, il faudra exécuter les lignes suivantes:


```python
import pickle
model = pickle.load( open("save-pickle-DDSRCNN-20200122", "rb"))
```

