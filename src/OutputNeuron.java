import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

public class OutputNeuron extends Neuron implements TargetNeuron {
	//immutable state
	private final Activation activationFunction;
	private final Set<Synapse> sourceSynapses;

	//mutable state
	private double target;
	private double error;

	//constructors
	public OutputNeuron(Activation function) {
		activationFunction = Objects.requireNonNull(function, "Invalid null function.");
		sourceSynapses = new HashSet();
	}

	//output neuron behaviour
	public void setTarget(double target) {
		this.target = target;
	}

	public double getError() {
		return error;
	}

	//target neuron behaviour
	@Override
	public double computeForwardSignal() {
		input = 0;
		sourceSynapses.forEach(synapse -> input += synapse.computeForwardSignal());
		return output = activationFunction.computeFunction(input);
	}

	@Override
	public double computeBackwardSignal() {
		return gradient = (error = target - output) * activationFunction.computeDerivative(input);
	}

	@Override
	public void setSourceSynapse(Synapse synapse) {
		sourceSynapses.add(Objects.requireNonNull(synapse, "Invalid null synapse."));
	}
}
