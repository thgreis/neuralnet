public class LinearActivation extends Activation {
	//activation behaviour
	@Override
	public double computeFunction(double input) {
		return input;
	}

	@Override
	public double computeDerivative(double input) {
		return 1.0;
	}
}
