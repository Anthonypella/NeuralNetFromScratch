using System;
using System.Collections.Generic;
using System.Threading;

namespace NeuralNet1
{

    class Program
    {
        static void Main(string[] args)
        {
            DateTime start = DateTime.Now;
            Random r = new Random();
            int[] dimensions = {10,1};
            manager manage = new manager(dimensions);
            double[] constants = manage.generateConstants(dimensions[0], 100);
            for (int k = 0; k <1000000; k++)
            {
                //one batch
                
                double[,] batchVals = new double[dimensions[0]+1, 10];

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
                   //Console.WriteLine( manage.getLoss(goal));
                    double[] properties = manage.getDerivatesFromInput(goal);

                    for (int l = 0; l < batchVals.GetLength(0)-1; l++)
                    {
                        batchVals[l, i] = properties[l];
                    }
                    batchVals[batchVals.GetLength(0) - 1, i] = properties[properties.Length - 1];
                }

                manage.updateParameters(manage.trainingBatch(batchVals));
                //manage.printParams(dimensions[0]);
                if (manage.checkAccuracy(constants))
                {
                    manage.printParams(dimensions[0]);
                    Console.WriteLine("iterations ran = " + k);
                    Console.WriteLine("iterations * batch = " + k * batchVals.GetLength(1));
                    DateTime end = DateTime.Now;
                    Console.WriteLine(end - start);
                    return;
                    break;
                }
            }
            Console.WriteLine("Ran out of iteractions");

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
            for (int i = 0; i < topology.Count-1; i++) //loop through layers, last has no weights
            {
                for (int j = 0; j < topology[i].Count; j++) //loop through each layer
                {
                    topology[i][j].setWeightsRandom(topology[i + 1].Count); // creates # of random weights equal to next layer
                }
                
            }
        }
        public void forwardPropogate(double[] inputs)
        {
            for (int i = 0; i < topology[0].Count; i++)
            {
                topology[0][i].setInput(inputs[i]);
            }
            for (int i = 1; i < topology.Count; i++)
            {
                //loop through layer 1 - x
                for (int j = 0; j < topology[i].Count; j++)
                {
                    double[] w = getWeights(i-1, j);
                    double[] act = getActivation(i-1);
                    //for every output neuron get the weights of every neuron going into it
                    // weights of all nodes in prev layer
                    //activations of all nodes in previous layer                   
                    topology[i][j].calculateActivation(w, act);


                }
            }
            
        }

        public double[] getDerivatesFromInput(double goalValue)
        {
            double[] parameters = new double[topology[0].Count+1];
            double aL = topology[1][0].activation;
            double loss = Math.Pow(goalValue - aL, 2);
            for (int i = 0; i < topology[0].Count; i++)
            {
                //double weighti = topology[0][i].weights[0];
                double previ = topology[0][i].activation;
                double derCostToWeighti = 2 * previ * (goalValue - aL);
                parameters[i] = derCostToWeighti;
            }
            parameters[parameters.Length - 1] = 2 * (goalValue - aL);
            /*double weightX = topology[0][0].weights[0];
            double weightY = topology[0][1].weights[0];
            double aLPrevX = topology[0][0].activation;
            double aLPrevY = topology[0][1].activation;

            double derCostToWeightX = 2 * aLPrevX * (goalValue - aL); //der respect to weightX
            double derCostToWeightY = 2 * aLPrevY * (goalValue - aL); //der respect to weightY*/

            //double derCostToActivationX = 2 * weightX * (aL - goalValue); //unneeded for 2 layers
            //double derCostToActivationY = 2 * weightY * (aL - goalValue);
            return parameters;

        }

        public double[] trainingBatch(double[,] trainingData) //inputs one batch
        {
            double[] averages = new double[trainingData.GetLength(0)];
            //training data.length is number of parameters in a batch, [0].length is length of a batch
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
        public void updateParameters(double[] parameters)
        {
            for (int i = 0; i < parameters.Length-1; i++)
            {
                topology[0][i].weights[0] += parameters[i];
            }
            /*topology[0][0].weights[0] += parameters[0];
            topology[0][1].weights[0] += parameters[1];*/
            topology[1][0].bias += parameters[parameters.Length-1];
        }
        public double step(double val)
        {
            return val /(topology[0].Count * 10);
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
        public double[] generateConstants(int length,double range)
        {
            double[] dubs = new double[length];
            Random R = new Random();
            for (int i = 0; i < length-1; i++)
            {
                dubs[i] = R.NextDouble() * range;

            }
            dubs[length-1] = R.NextDouble() * range;
            return dubs;
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

    }



    public class neuron
    {
        public double activation;
        public double[] weights;
        public double bias;
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
                activation += ws[i] * activations[i];
            }
            activation += bias;
            //activation = sigmoid(activation);
        }
        public void reset()
        {
            activation = 0;
        }
        static double sigmoid(double val)
        {
            return Math.Exp(val) / (1 + Math.Exp(val));
        }

    }
    public class outputNeuron : neuron
    {
        public outputNeuron() { }
    }
    




}
