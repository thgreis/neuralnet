import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

public class BiasNeuron extends Neuron implements SourceNeuron {
	//immutable state
	private final Set<Synapse> targetSynapses;

	//constructors
	public BiasNeuron() {
		targetSynapses = new HashSet();
	}

	//source neuron behaviour
	@Override
	public double computeForwardSignal() {
		return output = input = 1.;
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
