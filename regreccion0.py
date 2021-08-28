#Alumno Juan Carlos Arredondo
import math

#Este programa se deberia ejecutar como python P2RegresionLineal.py


def csv():
    datosX = []
    datosY = []
    abrir=open("motor_emisiones.csv","r")
    line=abrir.readline()
    line=abrir.readline()
    while(line):
        datosX.append(float(line.split(",")[0]))
        datosY.append(float(line.split(",")[1]))
        line=abrir.readline()
    return datosX,datosY
#Al incluir instrucciones en los metodos de python podemos quitar la palabra reservada pass
#Todos los metodos que contienen la instruccion pass se deben sustituir por las operaciones correspondientes
datosX,datosY=csv()
#media
#Suma de todos los elementos entre el numero de elementos
#Numero de elementos de un arreglo = len(Arreglo)
def mu(valores):
	return sum(valores)/len(valores)
#Suma de los errores al cuadrado
def sse(reales, obtenidos):
	result = 0
	for i in range(len(reales)):
		result += pow(reales[i] - obtenidos[i], 2)
	return result

def mse(reales, obtenidos):
	return sse(reales, obtenidos)/len(reales)

def rmse(reales, obtenidos):
	return math.sqrt(mse(reales, obtenidos))

class RegresionLineal:
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
		self.a0 = None #None es el equivalente a null
		self.a1 = None

	def _sumxy(self):
		#Como debemos acceder a valores tanto de x como de y, iteramos por el indice
		result = 0
		for i in range(len(self.X)):
			result = result + self.X[i]*self.Y[i]
		return result

	def _sumx2(self):
		return pow(sum(self.X), 2)

	def _sumxI2(self):
		result = 0
		for i in range(len(self.X)):
			result += pow(self.X[i], 2)
		return result

	def _sumx(self):
		return sum(self.X)

	def _sumy(self):
		return sum(self.Y)

	def fit(self):
		#Metodo que calcula los parametros de la linea recta a0 y a1
		n = len(self.X) #Numero de puntos
		yPromedio = mu(self.Y)
		xPromedio = mu(self.X)
		a1Divisor = (n * self._sumxy() - (self._sumx() * self._sumy()))
		a1Dividendo = (n * self._sumxI2() - self._sumx2())
		self.a1 = a1Divisor / a1Dividendo
		print('a1', self.a1)
		self.a0 = yPromedio - self.a1 * xPromedio
		print('a0', self.a0)
		#deberia  ser algo asi: self.a0 = .....
		#Este metodo no retorna ningun valor solo calcula y almacena los valores en sus atributos
		pass

	def eval(self, valuesX):
		#metodo eval ya esta completa
		result = []
		if not self.a0:
			self.fit()
		#se hace una iteracion de los valores en valuesX
		#esta estructura de control es parecida a foreach en java
		#f(x) = a0+a1*x
		for value in valuesX:
			result.append(self.a0+(self.a1*value))
		return result
if __name__ == "__main__":
	rl = RegresionLineal(datosX, datosY)
	rl.fit()
	Y_ = rl.eval(datosX)
	print("mediaX: ", mu(datosX))
	print("mediaY: ", mu(datosY))
	print("Suma de los errores al cuadrado", sse(datosY, Y_))
	print("MSE", mse(datosY, Y_))
	print("RMSE", rmse(datosY, Y_))
