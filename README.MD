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
   - Abra o terminal (cmd) e dentro da pasta em que se encontra o download execute:
   ``` 
   python get-pip.py"
   ```

- Instalar o opencv contrib - https://pypi.org/project/opencv-contrib-python/
 ```
 pip install opencv-contrib-python
 ```
 
 Importante instalar somente o opencv-contrib-python e não o opencv.
 
 
 ##### COMO USAR
 
 Para rodar, no terminal, dentro da pasta do projeto executar:
 
  ```
 python pcaeigenfaces.py
  ```
 
 Também existe um arquivo no formato do jupyter notebook com o mesmo nome acima que pode ser usado.
 
 ##### SAÍDA EXEMPLO
 
 ```
Porcentagem de treinamento: 70.0%
Quantidade imagens de treino: 280
Quantidade imagens de teste: 120

10 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 651.6488631450829
Acurracia: 94.16666666666667
Acertos: 113

11 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 684.0322596241674
Acurracia: 95.83333333333334
Acertos: 115

12 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 706.5359788696995
Acurracia: 97.5
Acertos: 117

13 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 725.5290184444704
Acurracia: 97.5
Acertos: 117

14 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 745.006931376385
Acurracia: 96.66666666666667
Acertos: 116

15 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 766.2466268992425
Acurracia: 96.66666666666667
Acertos: 116

16 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 780.5887202064566
Acurracia: 96.66666666666667
Acertos: 116

17 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 796.2636122621825
Acurracia: 95.83333333333334
Acertos: 115

18 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 808.4725159458225
Acurracia: 95.0
Acertos: 114

19 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 825.9518777944432
Acurracia: 96.66666666666667
Acertos: 116

20 componentes principais
Distância mínima: 2.2250738585072014e-308
Distância máxima: 1.7976931348623157e+308
Distância média: 842.9670234649824
Acurracia: 96.66666666666667
Acertos: 116
```
