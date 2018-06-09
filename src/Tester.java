import java.util.*;

public class Tester {
	public static void main(String[] args) {
		double[][] xorData = {
			{0.0, 0.0, 0.0},
			{0.0, 1.0, 1.0},
			{1.0, 0.0, 1.0},
			{1.0, 1.0, 0.0}
		};

		int inputNeuronsNumber = 2;
		int outputNeuronsNumber = 1;
		int hiddenLayersNumber = 1;
		int hiddenNeuronsNumber = 3;
		double learningRate = .1;
		double momentumRate = .9;
		double meanSynapticWeight = .0;
		double standardDeviationSynapticWeight = .1;

		Activation activationFunction = new LogisticActivation();
		NeuralNetwork n1 = new NeuralNetwork(inputNeuronsNumber, outputNeuronsNumber, hiddenLayersNumber, hiddenNeuronsNumber, activationFunction, learningRate, momentumRate, meanSynapticWeight, standardDeviationSynapticWeight);

		double error = 0;
		double output = 0;

		for (int epoch = 1; epoch <= 6000; epoch++) {
			for (double[] pattern : xorData) {
				List<Double> inputs = new ArrayList<>();
				inputs.add(pattern[0]);
				inputs.add(pattern[1]);

				output = n1.forwardPropagate(inputs);
				error = n1.backwardPropagate(pattern[2]);

				/*
				n1.setInput(0, xorData[index][0]);
				n1.setInput(1, xorData[index][1]);
				n1.setTarget(0, xorData[index][2]);
				n1.train();
				*/
				//System.out.println("Output " + patternCounter + ": " + new java.text.DecimalFormat("###.#########").format(n1.getOutput(0)));
				//System.out.println("Error at pattern " + patternCounter + ": " + error);
				//for (int counter3 = 1; counter3 <= n1.neuronsLayers.size(); counter3++) {
				//	for (Neuron neuron : n1.neuronsLayers.get(counter3)) {
				//		String neuronOutput = new java.text.DecimalFormat("###.#####").format(neuron.getForwardSignal());
				//		//String neuronError = new java.text.DecimalFormat("###.#####").format(neuron.getErrorSignal());
				//		//System.out.println("Epoch: " + epochCounter + "\tPattern: " + patternCounter + "\tLayer: " + counter3 +  "\tNeuron output: " + neuronOutput /*+  "\tNeuron error: " + neuronError*/);
				//	}
				//}
				//if (epochCounter % 1000 == 0) {
				//	System.out.println("Epoch: "+ epochCounter + ", output: " + new java.text.DecimalFormat("###.#########").format(n1.getOutput(0)));
				//}
				//error = n1.computeError();
			}

			System.out.println("Error at epoch " + epoch + ":\t" + new java.text.DecimalFormat("###.#########").format(error));
			n1.reset();
		}
	}
}
