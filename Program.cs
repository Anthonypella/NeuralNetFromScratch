using System;
using System.Collections.Generic;
using System.Threading;
using MNIST.IO;
using System.Linq;
namespace NeuralNet1
{

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("hello world");
            DateTime start = DateTime.Now;
            Random r = new Random();
            int[] dimensions = { 784, 16, 16, 10 };
            manager manage = new manager(dimensions);

            string trainDataPath = "C:\\Users\\jason\\Desktop\\mnistdata\\train-images-idx3-ubyte.gz";
            string trainLabelPath = "C:\\Users\\jason\\Desktop\\mnistdata\\train-labels-idx1-ubyte.gz";
            string testDataPath = "C:\\Users\\jason\\Desktop\\mnistdata\\t10k-images-idx3-ubyte.gz";
            string testLabelPath = "C:\\Users\\jason\\Desktop\\mnistdata\\t10k-labels-idx1-ubyte.gz";

            var data = FileReaderMNIST.LoadImagesAndLables(trainLabelPath,trainDataPath);
            Console.WriteLine("hello");

            //TestCase[] testCases = data.ToArray();
            foreach (var number in data)
            {
                Console.WriteLine(number.Label);
                Console.WriteLine("hniggerrld");

            }

            //get input
            /*double[] input = new double[784];
            double[] goalVector = new double[10];

            int batchSize = 10;
            int numBatches = 10;


            for (int l = 0; l < numBatches; l++)
            {
                List<double[,]> gradientVectors = new List<double[,]>(); //each 2d array represents all the weights of the neurons on the layer, +1 for bias
                for (int i = 1; i < dimensions.Length - 1; i++)
                {
                    double[,] layerArray = new double[dimensions[i], dimensions[i - 1] + 1];
                    gradientVectors.Add(layerArray);
                }

                for (int j = 0; j < batchSize; j++)
                {
                    double[] propagatedDerOfLoss = manage.train(input, goalVector);
                    for (int i = dimensions.Length - 1; i > 1; i--) //run for every layer except the input (since input has no previous activations)
                    {
                        gradientVectors[i] = manage.add2dArray(gradientVectors[i], manage.getDerivativesOfWeightsAndBiasesForLayer(i, propagatedDerOfLoss)); //add weights from previous activation derivative
                        if (i >= 2) //don't apply to second to last layer, since there are no weights to update for the input layer
                        {
                            double[] propagatedDerOfLossResize = manage.getActivationDerivativeForHiddenLayer(i - 1, propagatedDerOfLoss); //calculate next propagation
                            Array.Resize<double>(ref propagatedDerOfLoss, propagatedDerOfLossResize.Length); //resize propagationDerOfLoss to match the previous layer
                            propagatedDerOfLoss = propagatedDerOfLossResize;
                        }
                    }
                }

                //average batch
                manage.averageBatchVals(ref gradientVectors, batchSize);
                manage.updateParameters(ref gradientVectors);
                
            }

            DateTime end = DateTime.Now;
            Console.WriteLine(end - start);



            /*Random r = new Random();
            int[] dimensions = {8,1};
            manager manage = new manager(dimensions);
            double[] constants = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
            Console.WriteLine("Hello World!");
            for (int k = 0; k <100000; k++)
            {
                //one batch
                
                double[,] batchVals = new double[dimensions[0]+1, 5];

                for (int i = 0; i < batchVals.GetLength(1); i++)
                {
                    double goal = 0;
                    double[] input = new double[dimensions[0]];
                    for (int j = 0; j < input.Length; j++)
                    {
                        input[j] = r.NextDouble() * 5;
                        goal += input[j] * constants[j];
                    }
                    goal += constants[constants.Length-1];
                    //Console.WriteLine("X = " + xVal);
                    //Console.WriteLine("Y = " + yVal);
                    manage.forwardPropogate(input);
                    manage.getLoss(goal);
                    double[] properties = manage.getDerivatesFromInput(goal);

                    for (int l = 0; l < batchVals.GetLength(0)-1; l++)
                    {
                        batchVals[l, i] = properties[l];
                    }
                    batchVals[batchVals.GetLength(0) - 1, i] = properties[properties.Length - 1];
                }

                manage.updateParameters(manage.trainingBatch(batchVals));
                manage.printParams(dimensions[0]);
                if (manage.checkAccuracy(constants))
                {
                    Console.WriteLine("iterations ran = " + k);
                    Console.WriteLine("iterations * batch = " + k * batchVals.GetLength(1));
                    DateTime end = DateTime.Now;
                    Console.WriteLine(end - start);
                    break;
                }
            }*/

        }
    }

    public class manager
    {
        List<List<neuron>> topology;
        manager() { }
        public manager(int[] neuronsPerLayer)
        {
            topology = new List<List<neuron>>(); //initialize empty network
            for (int i = 0; i < neuronsPerLayer.Length; i++)
            {
                topology.Add(new List<neuron>()); //generate empty layer
                for (int j = 0; j < neuronsPerLayer[i]; j++) //fills empty layer
                {
                    topology[i].Add(new neuron()); 
                }
            }
            initialize();
        }
        public void initialize()
        {

            for (int i = 1; i < topology.Count; i++) //loop through layers, last has no weights
            {
                for (int j = 0; j < topology[i].Count; j++) //loop through each layer
                {
                    topology[i][j].setWeightsRandom(topology[i - 1].Count); // creates # of random weights equal to next layer
                }
                
            }
        }


        public double[] train(double[] input, double[] goalVector)
        {
            forwardPropogate(input);
            double[] output = getDerOfLoss(goalVector);
            return output;
        }

        public void forwardPropogate(double[] input)
        {
            for (int i = 0; i < topology[0].Count; i++)
            {
                topology[0][i].setInput(input[i]);
            }
            for (int i = 1; i < topology.Count; i++)
            {
                //loop through layer 1 - x
                for (int j = 0; j < topology[i].Count; j++)
                {
                    double[] act = getActivation(i-1);
                    //for every output neuron get the weights of every neuron going into it
                    // weights of all nodes in prev layer
                    //activations of all nodes in previous layer                   
                    topology[i][j].calculateActivation(topology[i][j].weights, act);


                }
            }
            
        }

        public double[,] getDerivativesOfWeightsAndBiasesForLayer(int layer, double[] prevDerActivationVector) //could return a as well
        {
            double[,] weightNeuronMatrix = new double[topology[layer].Count, topology[layer][0].weights.Length+1];
            for (int i = 0; i < topology[layer].Count; i++)
            {
                for (int j = 0; j < topology[layer][i].weights.Length; j++)
                {
                    double dzdw = topology[layer - 1][j].activation;
                    double dadz = topology[layer][i].sigmoidPrime();
                    double dcda = prevDerActivationVector[i];
                    weightNeuronMatrix[i,j] = dzdw * dadz * dcda;
                }
                weightNeuronMatrix[i, topology[layer][0].weights.Length] = topology[layer][i].sigmoidPrime() * prevDerActivationVector[i];
            }
            return weightNeuronMatrix;
        }

        public double[] getDerOfLoss(double[] goalVector) //calculate output layer's activation derivatives
        {
            int outputLayerLength = topology[topology.Count - 1].Count;
            double[] derActivationVector = new double[outputLayerLength];
            for (int i = 0; i < outputLayerLength; i++)
            {
                derActivationVector[i] = 2 * (topology[outputLayerLength][i].activation - goalVector[i]);

            }
            return derActivationVector;
        }

        public double[] getActivationDerivativeForHiddenLayer(int layerNum, double[] prevDerActivationVector)
        {
            int lengthOfForwardLayer = topology[layerNum + 1].Count;
            int lengthOfThisLayer = topology[layerNum].Count;
            double[] derActivationVector = new double[lengthOfThisLayer];

            for (int i = 0; i < lengthOfThisLayer; i++)
            {
                double sum = 0;
                for (int j = 0; j < lengthOfForwardLayer; j++)
                {
                    double dzda = topology[layerNum + 1][j].weights[i];
                    double dadz = topology[layerNum + 1][j].sigmoidPrime();
                    double dcdaL = prevDerActivationVector[j];
                    sum += dcdaL * dadz * dzda;
                }
                sum /= lengthOfForwardLayer;
                derActivationVector[i] = sum;
            }
            return derActivationVector;
        }
        public double[] trainingBatch(double[,] trainingData) //inputs one batch
        {
            double[] averages = new double[trainingData.GetLength(0)];
            //trainingData.GetLength(0) is number of parameters in a batch, trainingData.GetLength(1) is length of a batch
            for (int i = 0; i < trainingData.GetLength(0); i++)
            {
                double sum = 0;
                for (int j = 0; j < trainingData.GetLength(1); j++)
                {
                    sum += trainingData[i,j];
                }
                sum /= trainingData.GetLength(1);
                sum = step(sum);
                averages[i] = sum;
            }
            return averages;
        }

        public void averageBatchVals(ref List<double[,]> gradientVectors, int batchCount)
        {
            for (int i = 0; i < gradientVectors.Count; i++) //for each layer
            {
                for (int j = 0; j < gradientVectors[i].GetLength(0); j++) //loop through each neuron
                {
                    for (int k = 0; k < gradientVectors[i].GetLength(1); k++) //for each weight
                    {
                        gradientVectors[i][j, k] /= batchCount;
                        gradientVectors[i][j, k] = step(gradientVectors[i][j, k]);
                    }

                }
            }
        }

        public void updateParameters(ref List<double[,]> gradientVectors)
        {
            for (int i = 1; i < topology.Count; i++) //for each layer besides input
            {
                for (int j = 0; j < topology[i].Count; j++) //for each neuron in a layer
                {
                    for (int k = 0; k < topology[i][j].weights.Length; k++)
                    {
                        topology[i][j].weights[k] += gradientVectors[i][j, k];
                    }
                    topology[i][j].bias += gradientVectors[i][j, topology[i][j].weights.Length]; 
                }
            }
        }

        public double step(double val)
        {
            return val / 52.5;
        }
        public double getLoss(double goalValue)
        {
            double output = topology[1][0].activation;
            double loss = Math.Pow(goalValue - output, 2);
            //Console.WriteLine("Output of the net was: " + output);
            //Console.WriteLine("Goal value was: " + goalValue);
            //Console.WriteLine("Loss was: " + loss);
            return loss;
        }
        double[] getWeights(int layerNum,int targetNeuron)
        {
            double[] weights = new double[topology[layerNum].Count];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = topology[layerNum][i].weights[targetNeuron];
            }
            return weights;


        }
        double[] getActivation(int layerNum)
        {
            double[] act = new double[topology[layerNum].Count];
            for (int i = 0; i < act.Length; i++)
            {
                act[i] = topology[layerNum][i].activation;
            }
            return act;

        }
        public void printParams(int numOfParams)
        {
            for (int i = 0; i < numOfParams; i++)
            {
                Console.WriteLine(" Weight " + i + " = " + topology[0][i].weights[0]);
            }
            Console.WriteLine(" Bias = " + topology[1][0].bias);
        }

        public bool checkAccuracy(double[] goals)
        {
            for (int i = 0; i < goals.Length-1; i++)
            {
                if (Math.Abs(goals[i] - topology[0][i].weights[0]) > .00001)
                {
                    return false;
                }
            }
            if (Math.Abs(topology[1][0].bias - goals[goals.Length-1]) > .00001)
            {
                return false;
            }
            return true;
        }

        public static double[] addArray(double[] array1, double[] array2, int subtractLength = 0)
        {
            for (int i = 0; i < array1.Length - subtractLength; i++)
            {
                
                array1[i] += array2[i];
            }
            return array1;
        }

        public double[,] add2dArray(double[,] array1, double[,] array2)
        {
            for (int i = 0; i < array1.GetLength(0); i++)
            {
                for (int j = 0; j < array1.GetLength(1); j++)
                {
                    array1[i,j] += array2[i,j];
                }
            }
            return array1;
        }

    }



    public class neuron
    {
        public double activation;
        public double[] weights;
        public double bias;
        public double zValue;
        public neuron() { }
        public void setWeightsRandom(int numberOfWeights)
        {
            Random r = new Random();
            weights = new double[numberOfWeights];
            for (int i = 0; i < numberOfWeights; i++)
            {
                weights[i] = r.NextDouble();
            }
            bias = r.NextDouble();
        }   
        public void setInput(double input)
        {
            activation = input;
        }
        public void calculateActivation(double[] ws,double[] activations)
        {
            reset();
            for (int i = 0; i < ws.Length; i++)
            {
                zValue += ws[i] * activations[i];
            }
            zValue += bias;
            activation = sigmoid(zValue);
        }
        public void reset()
        {
            zValue = 0;
        }
        public double sigmoid(double val)
        {
            return 1 / (1 + Math.Exp(-val));
        }

        public double sigmoidPrime()
        {
            return sigmoid(zValue) * (1 - sigmoid(zValue));
        }

    }
    public class outputNeuron : neuron
    {
        public outputNeuron() { }
    }



}

