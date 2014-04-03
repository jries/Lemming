package org.lemming.inputs;
import java.util.Random;

public class RandomDouble extends SO<Double>{
	int N;
	double minVal;
	double maxVal;

	public RandomDouble(int N, double minVal, double maxVal){
		this.N = N;
		this.minVal = minVal;
		this.maxVal = maxVal;
	}
	@Override
	public boolean hasMoreOutputs() {
		return N>0;
	}

	@Override
	public Double newOutput() {
		Random rand = new Random();
    	N--;
    	return rand.nextDouble()*(maxVal-minVal) + minVal;
	}
}

