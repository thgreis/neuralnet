public class LogisticActivation extends Activation {
	//activation behaviour
	@Override
	public double computeFunction(double input) {
		return 1. / (1. + Math.exp(-input));
	}

	@Override
	public double computeDerivative(double input) {
		return computeFunction(input) * (1. - computeFunction(input));
	}
}
