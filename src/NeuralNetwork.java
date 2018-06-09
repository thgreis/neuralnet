import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

public class NeuralNetwork {
	//constant state
	private final static Random RANDONOMIZER = new Random();

	//immutable state
	private final Map<Integer, ArrayList<? extends Neuron>> layers;
	private final List<Synapse> synapses;
	private final Activation activationFunction;
	private final double learningRate;
	private final double momentumRate;
	private final int inputLayerIndex;
	private final int outputLayerIndex;
	private final double meanSynapticWeight;
	private final double standardDeviationSynapticWeight;

	//mutable state
	private double error;

	//constructors
	public NeuralNetwork(int inputNeurons, int outputNeurons, int hiddenLayers, int hiddenNeurons, Activation activationFunction, double learningRate, double momentumRate, double meanSynapticWeight, double standardDeviationSynapticWeight) {
		if (inputNeurons <= 0 || outputNeurons <= 0 || hiddenLayers < 0 || hiddenNeurons < 0 || learningRate <= 0 || learningRate > 1 || momentumRate < 0 || momentumRate >= 1) {
			throw new IllegalArgumentException("Invalid argument(s) values.");
		}

		this.activationFunction = Objects.requireNonNull(activationFunction, "Invalid null activation function.");
		this.learningRate = learningRate;
		this.momentumRate = momentumRate;
		this.inputLayerIndex = 1;
		this.outputLayerIndex = hiddenLayers + 2;
		this.layers = new HashMap<>();
		this.synapses = new ArrayList<>(/*(int)inputNeurons * (hiddenNeurons * hiddenLayers)*/);
		this.meanSynapticWeight = meanSynapticWeight;
		this.standardDeviationSynapticWeight = standardDeviationSynapticWeight;

		//adicionando camada e neuronios de entrada
		ArrayList inputNeuronsList = new ArrayList<>(inputNeurons);
		for (int neuronCounter = 1; neuronCounter <= inputNeurons; neuronCounter++) {
			inputNeuronsList.add(new InputNeuron());
		}

		inputNeuronsList.add(new BiasNeuron());
		layers.put(inputLayerIndex, inputNeuronsList);

		//adicionando camada e neuronios ocultos
		for (int layerCounter = inputLayerIndex + 1; layerCounter <= hiddenLayers + 1; layerCounter++) {
			ArrayList hiddenNeuronsList = new ArrayList<>(hiddenNeurons);

			for (int neuronCounter = 1; neuronCounter <= hiddenNeurons; neuronCounter++) {
				HiddenNeuron hiddenNeuron = new HiddenNeuron(this.activationFunction);
				hiddenNeuronsList.add(hiddenNeuron);

				for (Neuron neuron : layers.get(layerCounter - 1)) {
					synapses.add(new Synapse((SourceNeuron)neuron, hiddenNeuron, computeRandom(meanSynapticWeight, standardDeviationSynapticWeight), this.learningRate, this.momentumRate));
				}
			}

			hiddenNeuronsList.add(new BiasNeuron());
			layers.put(layerCounter, hiddenNeuronsList);
		}

		//adicionando camada e neuronios de saida
		ArrayList outputNeuronsList = new ArrayList<>(outputNeurons);
		for (int neuronCounter = 1; neuronCounter <= outputNeurons; neuronCounter++) {
			OutputNeuron outputNeuron = new OutputNeuron(this.activationFunction);
			outputNeuronsList.add(outputNeuron);

			for (Neuron neuron : layers.get(outputLayerIndex - 1)) {
				synapses.add(new Synapse((SourceNeuron)neuron, outputNeuron, computeRandom(meanSynapticWeight, standardDeviationSynapticWeight), this.learningRate, this.momentumRate));
			}
		}

		this.layers.put(hiddenLayers + 2, outputNeuronsList);
	}

	//neuralnetwork behaviour
	/*
	public void setInput(int inputNeuronIndex, double value) {
		if (inputNeuronIndex >= layers.get(inputLayerIndex).size()) {
			throw new IllegalArgumentException("Invalid input neuron index value.");
		}

		((InputNeuron)layers.get(inputLayerIndex).get(inputNeuronIndex)).setInput(value);
	}

	public void setTarget(int outputNeuronIndex, double value) {
		if (outputNeuronIndex >= layers.get(outputLayerIndex).size()) {
			throw new IllegalArgumentException("Invalid output neuron index value.");
		}

		((OutputNeuron)layers.get(outputLayerIndex).get(outputNeuronIndex)).setTarget(value);
	}

	public double getOutput(int outputNeuronIndex) {
		if (outputNeuronIndex >= layers.get(outputLayerIndex).size()) {
			throw new IllegalArgumentException("Invalid output neuron index value.");
		}

		return ((OutputNeuron)layers.get(outputLayerIndex).get(outputNeuronIndex)).getForwardSignal();
	}

	public void train() {
		//forward signal propagation
		for (Neuron neuron : layers.get(outputLayerIndex)) {
			neuron.computeForwardSignal();
		}

		//backward error propagation
		for (Neuron neuron : layers.get(inputLayerIndex)) {
			neuron.computeBackwardSignal();
		}

		//synaptic weights adjustment
		for (Synapse synapse : synapses) {
			synapse.computeSynapticWeight();
		}
	}

	public void rank() {
		//forward signal propagation
		for (Neuron neuron : layers.get(outputLayerIndex)) {
			neuron.computeForwardSignal();
		}
	}

	public void learn() {
		//backward error propagation
		for (Neuron neuron : layers.get(inputLayerIndex)) {
			neuron.computeBackwardSignal();
		}

		//weights adjustment
		for (Synapse synapse : synapses) {
			synapse.computeSynapticWeight();
		}
	}

	public double computeError() {
		for (OutputNeuron outputNeuron : (ArrayList)layers.get(outputLayerIndex)) {
			error += Math.pow(((OutputNeuron)outputNeuron).getError(), 2);
		}

		return 1. / 2. * error;
		//return 1. / 100 * error;
		//return Math.sqrt(error / 4);
	}
	*/
	public double getError() {
		return 1d / 2d * error;
		//return 1. / 10 * error;
	}

	//ranker behaviour
	public double forwardPropagate(List<Double> pattern) {
		if (pattern.size() != layers.get(inputLayerIndex).size() - 1) {
			throw new IllegalArgumentException("Invalid input length.");
		}

		//setting neuralnet inputs
		for (int index = 0; index < pattern.size(); index++) {
			((InputNeuron)layers.get(inputLayerIndex).get(index)).setInput(pattern.get(index));
		}

		//forward signal propagation
		for (Neuron neuron : layers.get(outputLayerIndex)) {
			neuron.computeForwardSignal();
		}

		//getting neuralnet output
		return ((OutputNeuron)layers.get(outputLayerIndex).get(0)).getForwardSignal();
	}

	public double backwardPropagate(double target) {
		//setting target value
		((OutputNeuron)layers.get(outputLayerIndex).get(0)).setTarget(target);

		//backward error propagation
		for (Neuron neuron : layers.get(inputLayerIndex)) {
			neuron.computeBackwardSignal();
		}

		//synaptic weights adjustment
		for (Synapse synapse : synapses) {
			synapse.computeSynapticWeight();
		}

		//computing neuralnet error
		for (OutputNeuron outputNeuron : (ArrayList<OutputNeuron>)layers.get(outputLayerIndex)) {
			error += Math.pow(((OutputNeuron)outputNeuron).getError(), 2);
		}

		return 1d / 2d * error;
	}

	//resettable behaviour
	public void reset() {
		//for (Synapse synapse : synapses) {
		//	synapse.setSynapticWeight(computeRandom(meanSynapticWeight, standardDeviationSynapticWeight));
		//}

		error = 0;
	}

	//miscellaneous
	private static double computeRandom(double mean, double standardDeviation) {
		return RANDONOMIZER.nextGaussian() * standardDeviation + mean;
	}
}
