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
            for (int i = 0; i < 100; i++)
            {
                double xVal = r.NextDouble() * 5;
                double yVal = r.NextDouble() * 5;
                double goal = xVal * 3.0 + yVal * 2.0;
                double[] input = { xVal, yVal };
                Console.WriteLine("X = " + xVal);
                Console.WriteLine("Y = " + yVal);
                manage.forwardPropogate(input);
                manage.getLoss(goal);
            }

        }
    }

    public class manager
    {
        List<List<neuron>> topology;
        manager() { }
        public manager(int[] neuronsAmounts)
        {
            topology = new List<List<neuron>>();
            for (int i = 0; i < neuronsAmounts.Length; i++)
            {
                topology.Add(new List<neuron>());
                for (int j = 0; j < neuronsAmounts[i]; j++)
                {
                    topology[i].Add(new neuron());
                }
            }
            initialize();
        }
        public void initialize()
        {
            for (int i = 0; i < topology.Count-1; i++)
            {
                for (int j = 0; j < topology[i].Count; j++)
                {
                    topology[i][j].setWeightsRandom(topology[i + 1].Count); // make fully connected
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
        public double getLoss(double goalValue)
        {
            double output = topology[1][0].activation;
            double loss = Math.Pow(goalValue - output, 2);
            Console.WriteLine("Output of the net was: " + output);
            Console.WriteLine("Goal value was: " + goalValue);
            Console.WriteLine("Loss was: " + loss);
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
            activation = sigmoid(activation);
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
    
    public  struct edge
    {
        public double weight;
        public neuron forwardConnection;
        public neuron backConnection;
    }




}
