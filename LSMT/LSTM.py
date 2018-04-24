#https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
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
	return datetime.strptime('199'+x, '%Y-%m')

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

# ajustar uma rede LSTM aos dados de treinamento
def ajustarLSTM(train, batch_size, nb_epoch, neuronios):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	modelo = Sequential()
	modelo.add(LSTM(neuronios, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	modelo.add(Dense(1))
	modelo.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		modelo.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		modelo.reset_states()
	return modelo

# fazer previsão de um passo
def previsaoLSTM(modelo, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = modelo.predict(X, batch_size=batch_size)
	return yhat[0,0]

def previsaoATM1():
        # carregar dados
        series = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
        n_test = -12
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
        # ajustar modelo
        modeloLSTM = ajustarLSTM(trainNormalizado, 1, 3000, 4)
        
        # prever todo o conjunto de dados de treinamento para aumentar o estado de previsão
        trainRemodelado = trainNormalizado[:, 0].reshape(len(trainNormalizado), 1, 1)
        modeloLSTM.predict(trainRemodelado, batch_size=1)
        
        # walk-forward validation nos dados de teste
        previsoes = list()
        for i in range(len(testNormalizado)):
	        # fazer previsão de um passo
	        X, y = testNormalizado[i, 0:-1], testNormalizado[i, -1]
	        print(X)
	        yhat = previsaoLSTM(modeloLSTM, 1, X)
	        # inverter escala
	        yhat = inverteNormalizar(escala, X, yhat)
	        # inverter diferenca
	        yhat = InverteDiferenca(valorLinhas, yhat, len(testNormalizado)+1-i)
                                
	        previsoes.append(yhat)
	        expected = valorLinhas[len(train) + i + 1]
	        #print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
        
        # relatar desempenho
        rmse = sqrt(mean_squared_error(valorLinhas[n_test:], previsoes))
        
        #print('Test RMSE: %.3f' % rmse)
        # plot observado vs predito
        pyplot.plot(valorLinhas[n_test:])
        pyplot.plot(previsoes)
        pyplot.show()

t = timeit.Timer("previsaoATM1()", "from __main__ import previsaoATM1")
print (t.repeat())
