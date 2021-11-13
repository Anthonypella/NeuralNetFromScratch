using System;
using System.Collections.Generic;


namespace NeuralNet1
{

    class Program
    {
        static void Main(string[] args)
        {
            Random r = new Random();
            int[] dimensions = {2,1};
            manager manage = new manager(dimensions);
            Console.WriteLine("Hello World!");
            for (int k = 0; k <20000; k++)
            {
                //one batch
                double[,] batchVals = new double[3, 100];
                for (int i = 0; i < 100; i++)
                {
                    double xVal = r.NextDouble() * 5;
                    double yVal = r.NextDouble() * 5;
                    double goal = xVal * 3 + yVal * 2; //define equation
                    double[] input = { xVal, yVal };
                    //Console.WriteLine("X = " + xVal);
                    //Console.WriteLine("Y = " + yVal);
                    manage.forwardPropogate(input);
                    manage.getLoss(goal);
                    double[] properties = manage.getDerivatesFromInput(goal);
                    batchVals[0, i] = properties[0];
                    batchVals[1, i] = properties[1];
                    batchVals[2, i] = properties[2];
                }

                manage.updateParameters(manage.trainingBatch(batchVals));
                manage.printParams();
            }


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
            if(topology.Count == 2)
            {
                for (int i = 0; i < topology[0].Count; i++)
                {
                    topology[0][i].setInput(inputs[i]);
                }
                for (int i = 0; i < topology[1].Count; i++)
                {
                    double[] w = getWeights(0, i);
                    double[] act = getActivation(0);
                    //for every output neuron get the weights of every neuron going into it
                    // weights of all nodes in prev layer
                   //activations of all nodes in previous layer                   
                    topology[1][i].calculateActivation(w, act);
                }
            }
            else
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
                        topology[j][i].calculateActivation(w, act);


                    }
                }
            }
        }

        public double[] getDerivatesFromInput(double goalValue)
        {
            double aL = topology[1][0].activation;
            double loss = Math.Pow(goalValue - aL, 2);

            double weightX = topology[0][0].weights[0];
            double weightY = topology[0][1].weights[0];
            double aLPrevX = topology[0][0].activation;
            double aLPrevY = topology[0][1].activation;

            double derCostToWeightX = 2 * aLPrevX * (goalValue - aL); //der respect to weightX
            double derCostToWeightY = 2 * aLPrevY * (goalValue - aL); //der respect to weightY

            //double derCostToActivationX = 2 * weightX * (aL - goalValue); //unneeded for 2 layers
            //double derCostToActivationY = 2 * weightY * (aL - goalValue);

            double derCostToBias = 2 * (goalValue - aL);

            double[] myArray = { derCostToWeightX, derCostToWeightY, derCostToBias };
            return myArray;

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
            topology[0][0].weights[0] += parameters[0];
            topology[0][1].weights[0] += parameters[1];
            topology[1][0].bias += parameters[2];
        }
        public double step(double val)
        {
            return val / 100;
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
        public void printParams()
        {
            Console.WriteLine("X weight: " + topology[0][0].weights[0]);
            Console.WriteLine("Y weight: " + topology[0][1].weights[0]);
            Console.WriteLine("Bias: " + topology[1][0].bias);
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
