#include "stdafx.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <omp.h>
#include <math.h>

using namespace std;

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void)
	{
		return m_trainingDataFile.eof();
	}
	void getTopology(vector<unsigned> &topology);

	// Devuelve el número de valores de entrada leídos del archivo:
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
	inputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
	targetOutputVals.clear();
	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);
	string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}
	return targetOutputVals.size();
}

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer, int thread_count);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer &nextLayer, int thread_count);
	void updateInputWeights(Layer &prevLayer,int thread_count);
private:
	static double eta; // [0.0...1.0] tasa neta de entrenamiento global
	static double alpha; // [0.0...n] multiplicador del último cambio de peso[momentum]
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	// randomWeight: 0 - 1
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer, int thread_count) const;
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};

double Neuron::eta = 0.15; // tasa neta de aprendizaje global
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]

void Neuron::updateInputWeights(Layer &prevLayer,int thread_count)
{
	// Los pesos que se actualizarán están en el contenedor de conexión
	// en los nuerons en la capa anterior
	int n; double oldDeltaWeight, newDeltaWeight;
	//#pragma omp parallel for private(n)  num_threads(8)
//#pragma omp parallel  num_threads(thread_count)
	for (n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		newDeltaWeight =
			// Entrada individual, magnificada por el gradiente y la velocidad del tren:
			eta
			* neuron.getOutputVal()
			* m_gradient
			// También agregue ímpetu = una fracción del peso del delta anterior
			+ alpha
			* oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
double Neuron::sumDOW(const Layer &nextLayer,int thread_count) const
{
	double sum = 0.0; int n;
	// Suma nuestras contribuciones de los errores en los nodos que alimentamos
	#pragma omp parallel for private(n)  num_threads(thread_count)
	for ( n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer, int thread_count)
{
	//#pragma omp parallel  num_threads(thread_count)
	double dow = sumDOW(nextLayer, thread_count);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}
void Neuron::calcOutputGradients(double targetVals)
{
	double delta = targetVals - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer, int thread_count)
{
	int n;
	// Suma los resultados de la capa anterior (que son nuestras entradas)
	// Incluye el nodo de sesgo de la capa anterior.
	double start;
	double end;
	int sum = 0;
		#pragma omp parallel for num_threads(thread_count)
		for ( n = 0; n < prevLayer.size(); ++n)
		{
   			sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
		}
	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}
// ****************** class Net ******************
class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals,int thread_count);
	void backProp(const vector<double> &targetVals,int thread_count);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Número de muestras de entrenamiento para promediar durante

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<double> &targetVals, int thread_count)
{
	// Calcular error neto general (RMS de errores de neurona de salida)

	Layer &outputLayer = m_layers.back();
	m_error = 0.0; int n; int layerNum; double delta;
	//#pragma omp parallel  num_threads(thread_count)
	for (n = 0; n < outputLayer.size() - 1; ++n)
	{
		delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta *delta;
	}
	m_error /= outputLayer.size() - 1; // obtener error promedio al cuadrado
	m_error = sqrt(m_error); // RMS

// Implementar una medición promedio reciente

	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);
	// Calcular gradientes de capa de salida
	
	for (int n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calcule gradientes en capas ocultas

	for ( layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer, thread_count);
		}
	}

	// Para todas las capas, desde las salidas hasta la primera capa oculta,
	// actualizar los pesos de conexión
	
	for (layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];
		for (n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer, thread_count);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals,int thread_count)
{
	// Compruebe el numero de  inputVals igual a  bias esperado
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Asignar {pestillo} los valores de entrada en las neuronas de entrada
	//#pragma omp parallel for  num_threads(thread_count)
	for (int i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	int layerNum, n;
	// Progpagacion hacia adelante
	//#pragma omp parallel num_threads(thread_count)
	//#pragma omp for
	for (layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer, thread_count);
		}
 	}
}
Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	int layerNum, neuronNum;
	#pragma paralell for num_threads(4)
	for (layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		// numOutputs de la capa [i] es el numInputs de la capa [i + 1]
		// la salida numérica de la última capa es 0
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		// Hemos creado una nueva Capa, ahora la llenamos con neuronas, y
		// agrega una neurona de sesgo a la capa:
		for ( neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			//cout << "Mad a Neuron!" << endl;
		}
		// Forzar el valor de salida del nodo de polarización a 1.0. Es la última neurona creada arriba
		m_layers.back().back().setOutputVal(1.0);
	}
}

void showVectorVals(string label, vector<double> &v)
{
	//cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		//cout << v[i] << " ";
	}
	//cout <<   endl;
}

int main()
{
	int thread_count=4, thread_count1=1;
	double iter = 1,end,start;
	
	for (int i = 0; i <= 2; i++)
	{
	thread_count = pow(2,i);
	TrainingData trainData("trainingData5.txt");
	//e.g., {3, 2, 1 }
	vector<unsigned> topology;
	//topology.push_back(3);
	//topology.push_back(2);
	//topology.push_back(1);
	trainData.getTopology(topology);
	Net myNet(topology);	
	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
		start = omp_get_wtime();
		while (!trainData.isEof())
	//#pragma omp parallel for num_threads(thread_count)
		//for (int i = 1; i <= 30000; i++)
		{
			++trainingPass;
			//cout << endl <<"Pasar" << trainingPass;
			// Obtener nuevos datos de entrada y alimentarlos hacia adelante:
			if (trainData.getNextInputs(inputVals) != topology[0])
				break;
			showVectorVals(": Inputs :", inputVals);
			myNet.feedForward(inputVals, thread_count);

			// Recoge los resultados reales de la red
			myNet.getResults(resultVals);
			showVectorVals("Outputs:", resultVals);

			// Entrenar a la red cuáles deberían haber sido los resultados :
			trainData.getTargetOutputs(targetVals);
			showVectorVals("Objetivos:", targetVals);
			//cout << targetVals.size() << "target" << endl;
			//cout << topology.back() << "back" << endl;
			assert(targetVals.size() == topology.back());
			myNet.backProp(targetVals, thread_count);
			// Informe qué tan bien está funcionando la capacitación, promedio sobre recnet
			//	cout << "Error promedio reciente neto : "
				//<< myNet.getRecentAverageError() << endl;
		}
		end = omp_get_wtime();
		double efic, speedup;
		if (thread_count == 1) {
			iter = end - start;
		}
		if (thread_count == 4) {
		//	end = end - 0.4;
		}
		speedup = iter / (end - start);
		efic = speedup / thread_count;
		cout << "=============================================" << endl;
		cout << end - start << "tiempo"<<endl;
		cout << thread_count << " hilos" << endl;
		cout << speedup << " SpeedUp" << endl;
		cout << efic << " Eficiencia" << endl;
	}
	system("pause");
	return 0;
}




