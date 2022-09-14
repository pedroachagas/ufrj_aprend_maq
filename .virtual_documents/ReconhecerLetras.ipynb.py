import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import HTML
import seaborn as sns

from matplotlib.image import imread 
from PIL import Image

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import cluster


def jupyter_settings():
    get_ipython().run_line_magic("matplotlib", " inline")

    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24

    display(HTML('<style>.conteiner{width:100% get_ipython().getoutput("important;}</style>'))")

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option('display.expand_frame_repr', False)
    # configura o pandas para quantidade de casas decimeis
    pd.set_option('display.float_format', lambda x: 'get_ipython().run_line_magic(".2f'", " % x)")

    sns.set()
jupyter_settings()


path_of_the_directory= 'Img/'
file_names = []
for filename in os.listdir(path_of_the_directory):
    f = os.path.join(path_of_the_directory,filename)
    if os.path.isfile(f):
        file_names.append(f)
file_names.sort()
file_names


labels = pd.read_csv('english.csv')
labels.sample(10)


import cv2

image_data = {}
for image in file_names:
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # The image pixels have range [0, 255] # Now the pixels have range [0, 1] 
    resized = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    img_list = np.array(resized.tolist()).flatten() # We have a list of lists of pixellens
    image_data[image] = img_list

df= pd.DataFrame(image_data)



df_T = df.T.reset_index()
df_T


letter_data = labels.merge(df_T, how='inner', left_on='image', right_on='index')


conjuntoDados = letter_data.drop(columns='image')


#1. Quantos exemplos possui
conjuntoDados['label'].count()


#2. Quantos exemplos cada classe possui
conjuntoDados.label.value_counts().plot(kind='bar');


pd.isnull(conjuntoDados).count(True)


#3. Visualizando os dígitos

#pegando um exemplo de cada dígito
exemplos = {}

for digito in range(len(conjuntoDados.labels)):
    for indice,linha in conjuntoDados.iterrows():
        if digito == linha['label']:
            vetorImg = np.zeros((1,784))
            j = 0
            for pixel in linha.keys()[1:]:
                vetorImg[0,j] = linha[pixel]
                j+=1
            matrizImg = vetorImg.reshape((28,28))
            break
    exemplos[digito] = matrizImg

#gerando imagem de cada exemplo
for digito in range(10):
    plt.subplot(2,5, digito+1)
    plt.axis('off')
    plt.imshow(exemplos[digito], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Dígito = get_ipython().run_line_magic("d'%int(digito))", "")
plt.show()


classificador = svm.SVC(kernel='linear')


#Olhar a quantidade de dados e separar 80% para treino e 20% para teste 
tamanho = conjuntoDados['label'].count()
tamanhoTreino = int(tamanho*0.8) #80% para treino
tamanhoTeste = tamanho-tamanhoTreino
print('Tamanho do treino:',tamanhoTreino)
print('Tamanho do teste:',tamanhoTeste)

#Separar conjunto de treino e conjunto de teste
treinoFeatures,treinoClasses = conjuntoDados[conjuntoDados.columns[1:]].iloc[:tamanhoTreino],conjuntoDados[0:tamanhoTreino]['label'] 
testeFeatures,testeClasses = conjuntoDados[conjuntoDados.columns[1:]
                                          ].iloc[tamanhoTreino:],conjuntoDados[tamanhoTreino:]['label']


#Treina o classificador com os dados de treino
classificador.fit(treinoFeatures,treinoClasses)
testeFeatures.head()



#Usa o modelo criado para fazer a predição das classes para exemplos de teste
testePrevisao = classificador.predict(testeFeatures)


#Observa o resultado gerado pelo classificador
matrizConfusao = metrics.confusion_matrix(testeClasses,testePrevisao)
sns.heatmap(matrizConfusao, annot=True)


#Relatório de métricas
metricas = metrics.classification_report(testeClasses,testePrevisao)
print(metricas)


estimador = cluster.KMeans(n_clusters=10)

treinoFeatures = conjuntoDados.iloc[:,1:]
clusters = estimador.fit_predict(treinoFeatures)
estimador.cluster_centers_.shape


fig = plt.figure(figsize=(12,5))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.imshow(estimador.cluster_centers_[i].reshape((28, 28)), cmap=plt.cm.binary)
fig
