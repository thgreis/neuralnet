import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

public class HiddenNeuron extends Neuron implements SourceNeuron, TargetNeuron {
	//immutable state
	private final Activation activationFunction;
	private final Set<Synapse> sourceSynapses;
	private final Set<Synapse> targetSynapses;

	//constructors
	public HiddenNeuron(Activation function) {
		activationFunction = Objects.requireNonNull(function, "Invalid null function.");
		sourceSynapses = new HashSet<>();
		targetSynapses = new HashSet<>();
	}

	//source/target neuron behaviour
	@Override
	public double computeForwardSignal() {
		input = 0;
		sourceSynapses.forEach(synapse -> input += synapse.computeForwardSignal());
		return output = activationFunction.computeFunction(input);
	}

	@Override
	public double computeBackwardSignal() {
		gradient = 0;
		targetSynapses.forEach(synapse -> gradient += synapse.computeBackwardSignal());
		return gradient = gradient * activationFunction.computeDerivative(input);
	}

	@Override
	public void setTargetSynapse(Synapse synapse) {
		targetSynapses.add(Objects.requireNonNull(synapse, "Invalid null synapse."));
	}

	@Override
	public void setSourceSynapse(Synapse synapse) {
		sourceSynapses.add(Objects.requireNonNull(synapse, "Invalid null synapse."));
	}
}
