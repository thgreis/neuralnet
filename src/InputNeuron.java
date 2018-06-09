import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

public class InputNeuron extends Neuron implements SourceNeuron {
	//immutable state
	private final Set<Synapse> targetSynapses;

	//constructors
	public InputNeuron() {
		targetSynapses = new HashSet();
	}

	//input neuron behaviour
	public void setInput(double input) {
		this.input = input;
	}

	//source neuron behaviour
	@Override
	public double computeForwardSignal() {
		return output = input;
	}

	@Override
	public double computeBackwardSignal() {
		targetSynapses.forEach(synapse -> synapse.computeBackwardSignal());
		return gradient = 0;
	}

	@Override
	public void setTargetSynapse(Synapse synapse) {
		targetSynapses.add(Objects.requireNonNull(synapse, "Invalid null synapse."));
	}
}
