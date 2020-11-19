### Projeto exemplo fazendo uso de opencv e eigenfaces em python.

#### Disciplina de aprendizado não supervizionado da Universidade Regional de Blumenau.

O objeto do trabalho é desenvolver e aplicar a Análise de Componentes Principais (PCA) para fazer um algoritmo que faça o reconhecimento visual de uma base imagens.
A base de dados base a ser usada é a (https://github.com/lobokoch/unsupervised-learning/blob/main/dataset/ORL.rar).
Esse projeto ainda conta com mais 10 imagens pessoais, extras, para ttestar o algoritmo.
Como amostragem, foi utilizado um holdout de 70% das imagens para treinamento de 30% para testes.

##### REQUISITOS

Python, pip, opencv-contrib-python 

- Instalar Python 3.8 - https://www.python.org/downloads/
   (escolher a ultima versão e instalar através do executável)
   
- Instalar pip para o python - https://phoenixnap.com/kb/install-pip-windows
   - Faça download do https://bootstrap.pypa.io/get-pip.py em uma pasta de sua escolha
   - Abra o terminal (cmd) e dentro da pasta em que se encontra o download execute python get-pip.py"
   
- Instalar o opencv contrib - https://pypi.org/project/opencv-contrib-python/
   - pip install opencv-contrib-python
   
   Importante instalar somente o opencv-contrib-python e não o opencv.
 
 
 
 ##### COMO USAR
 
 Para rodar, no terminal, dentro da pasta do projeto executar python "pcaeigenfaces.py";
 
 Também existe um arquivo no formato do jupyter notebook com o mesmo nome acima que pode ser usado.
 
 ##### SAÍDA EXEMPLO
 
 ```
Porcentagem de treinamento: 70.0%
Quantidade imagens de treino: 288
Quantidade imagens de teste: 123
-------------------
10 componentes principais
Distância mínima: 127.60090744090013
Distância máxima: 2074.9416708859344
Distância média: 681.9834613954098
Acurracia: 92.68%
Acertos: 114
-------------------
11 componentes principais
Distância mínima: 130.62408638358733
Distância máxima: 2099.5313503365455
Distância média: 711.2151132536015
Acurracia: 92.68%
Acertos: 114
-------------------
12 componentes principais
Distância mínima: 132.05271541744548
Distância máxima: 2099.648096332083
Distância média: 735.2548744072348
Acurracia: 92.68%
Acertos: 114
-------------------
13 componentes principais
Distância mínima: 157.34505069242715
Distância máxima: 2109.9363047452957
Distância média: 761.193191977317
Acurracia: 93.50%
Acertos: 115
-------------------
14 componentes principais
Distância mínima: 159.06622050663583
Distância máxima: 2135.1632697778164
Distância média: 777.4622054988862
Acurracia: 93.50%
Acertos: 115
-------------------
15 componentes principais
Distância mínima: 166.80697411484414
Distância máxima: 2188.230278673134
Distância média: 800.5066794278644
Acurracia: 93.50%
Acertos: 115
-------------------
16 componentes principais
Distância mínima: 167.0910499854427
Distância máxima: 2224.8798922264677
Distância média: 818.256675334215
Acurracia: 91.87%
Acertos: 113
-------------------
17 componentes principais
Distância mínima: 169.01054709657174
Distância máxima: 2269.6201800760487
Distância média: 835.6439428824581
Acurracia: 92.68%
Acertos: 114
-------------------
18 componentes principais
Distância mínima: 180.97581874908948
Distância máxima: 2278.673936227622
Distância média: 847.6725529037762
Acurracia: 91.87%
Acertos: 113
-------------------
19 componentes principais
Distância mínima: 181.60173527025384
Distância máxima: 2309.323254012441
Distância média: 859.8645392338981
Acurracia: 92.68%
Acertos: 114
-------------------
20 componentes principais
Distância mínima: 184.0676341534224
Distância máxima: 2324.9775179347516
Distância média: 872.3160176264756
Acurracia: 93.50%
Acertos: 115
-------------------
```
