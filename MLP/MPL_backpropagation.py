from sklearn.neural_network import MLPRegressor
import random
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import numpy as np
from sklearn import preprocessing, cross_validation
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import timeit

# carregar dataset fazendo um paser em cada linha
def parser(x):
	return datetime.strptime('199'+x, '%Y-%m-%d')

# series temporais para supervisionada
def temporalParaSupervisionado(dado, lag=1):
	dtFrame = DataFrame(dado)
	colunas = [dtFrame.shift(i) for i in range(1, lag+1)]
	colunas.append(dtFrame)
	dtFrame = concat(colunas, axis=1)
	dtFrame.fillna(0, inplace=True)
	return dtFrame

# faz a diferenca entre um valor e seu antecessor
def diferenca(dado, intervalo=1):
	dif = list()
	for i in range(intervalo, len(dado)):
		valor = dado[i] - dado[i - intervalo]
		dif.append(valor)
	return Series(dif)

# inverte a diferenca
def InverteDiferenca(historia, yhat, intervalo=1):
	return yhat + historia[-intervalo]

# faz a normalizacao para [-1, 1]
def normalizar(train, test):
	# ajustar escala
	escala = MinMaxScaler(feature_range=(-1, 1))
	escala = escala.fit(train)
	# transformar treinamento
	train = train.reshape(train.shape[0], train.shape[1])
	trainNormalizado = escala.transform(train)
	# transformar teste
	test = test.reshape(test.shape[0], test.shape[1])
	testNormalizado = escala.transform(test)
	return escala, trainNormalizado, testNormalizado

# inverter normalizacao
def inverteNormalizar(escala, X, valor):
	novaLinha = [x for x in X] + [valor]
	array = numpy.array(novaLinha)
	array = array.reshape(1, len(array))
	invertido = escala.inverse_transform(array)
	return invertido[0, -1]


#create the model.
regressor=MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,random_state=0)

series = read_csv('BigStreet.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
n_test = -360
# transformar dados para estacionários
valorLinhas = series.values
valorDif = diferenca(valorLinhas, 1)

# transformtar series temporais em supervisionadas
dadoSupervisionado = temporalParaSupervisionado(valorDif, 1)
dadoSupervisionadoValoes = dadoSupervisionado.values

# separar dos dados em treinamento e teste
train, test = dadoSupervisionadoValoes[0:n_test], dadoSupervisionadoValoes[n_test:]

# normalizar dados
escala, trainNormalizado, testNormalizado = normalizar(train, test)

train_X, train_y = trainNormalizado[:, 0:-1], trainNormalizado[:, -1]
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1])

regressor.fit(train_X, train_y)

# walk-forward validation nos dados de teste
previsoes = list()
for i in range(len(testNormalizado)):
	# fazer previsão de um passo
	x, y = testNormalizado[i, 0:-1], testNormalizado[i, -1]
	
	X = x.reshape(1, len(x))
	yhat = regressor.predict(X)
	# inverter escala
	yhat = inverteNormalizar(escala, x, yhat)
	# inverter diferenca
	yhat = InverteDiferenca(valorLinhas, yhat, len(testNormalizado)+1-i)
                        
	previsoes.append(yhat)
	expected = valorLinhas[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# relatar desempenho
rmse = sqrt(mean_squared_error(valorLinhas[n_test:], previsoes))

print('Test RMSE: %.3f' % rmse)
# plot observado vs predito
pyplot.plot(valorLinhas[n_test:])
pyplot.plot(previsoes)
pyplot.show()

print(yhat)

