public class TanhActivation extends Activation {
	//activation behaviour
	@Override
	public double computeFunction(double input) {
		return Math.tanh(input);
	}

	@Override
	public double computeDerivative(double input) {
		return 1.0 - Math.pow(computeFunction(input), 2.0);
	}
}
