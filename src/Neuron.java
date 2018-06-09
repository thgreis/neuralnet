public abstract class Neuron implements Transmitter {
	//mutable state
	protected double input;
	protected double output;
	protected double gradient;

	//neuron behaviour
	@Override
	public double getForwardSignal() {
		return output;
	}

	@Override
	public double getBackwardSignal() {
		return gradient;
	}
}
