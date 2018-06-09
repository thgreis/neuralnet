import java.util.Objects;

public class Synapse implements Transmitter {
	//immutable state
	private final SourceNeuron preSynapticNeuron;
	private final TargetNeuron postSynapticNeuron;
	private final double learningRate;
	private final double momentumRate;

	//mutable state
	private double synapticWeight;
	private double output;
	private double error;
	private double delta;

	//constructors
	public Synapse(SourceNeuron preSynapticNeuron, TargetNeuron postSynapticNeuron, double synapticWeight, double learningRate) {
		this(preSynapticNeuron, postSynapticNeuron, synapticWeight, learningRate, .0);
	}

	@SuppressWarnings("LeakingThisInConstructor")
	public Synapse(SourceNeuron preSynapticNeuron, TargetNeuron postSynapticNeuron, double synapticWeight, double learningRate, double momentumRate) {
		if (learningRate <= 0 || learningRate > 1 || momentumRate < 0 || momentumRate >= 1) {
			throw new IllegalArgumentException("Invalid argument(s) values.");
		}

		this.preSynapticNeuron = Objects.requireNonNull(preSynapticNeuron, "Invalid null pre synaptic neuron.");
		this.postSynapticNeuron = Objects.requireNonNull(postSynapticNeuron, "Invalid null post synaptic neuron.");
		this.preSynapticNeuron.setTargetSynapse(this);
		this.postSynapticNeuron.setSourceSynapse(this);
		this.learningRate = learningRate;
		this.momentumRate = momentumRate;
		this.synapticWeight = synapticWeight;
	}

	//synapse behaviour
	//public double computeSynapticWeight(boolean batch, boolean update) {
	public double computeSynapticWeight() {
		/*
		if (batch) {
			if (update) {
				return delta += momentumRate * delta + learningRate * postSynapticNeuron.getBackwardSignal() * preSynapticNeuron.getForwardSignal();
			} else {
				return synapticWeight += delta += momentumRate * delta + learningRate * postSynapticNeuron.getBackwardSignal() * preSynapticNeuron.getForwardSignal();
			}
		} else {
			return synapticWeight += delta = momentumRate * delta + learningRate * postSynapticNeuron.getBackwardSignal() * preSynapticNeuron.getForwardSignal();
		}
		*/
		return synapticWeight += delta = momentumRate * delta + learningRate * postSynapticNeuron.getBackwardSignal() * preSynapticNeuron.getForwardSignal();
	}

	public double getSynapticWeight() {
		return synapticWeight;
	}

	protected void setSynapticWeight(double synapticWeight) {
		this.synapticWeight = synapticWeight;
	}

	//transmitter behaviour
	@Override
	public double computeForwardSignal() {
		return output = preSynapticNeuron.computeForwardSignal() * synapticWeight;
	}

	@Override
	public double computeBackwardSignal() {
		return error = postSynapticNeuron.computeBackwardSignal() * synapticWeight;
	}

	@Override
	public double getForwardSignal() {
		return output;
	}

	@Override
	public double getBackwardSignal() {
		return error;
	}
}
