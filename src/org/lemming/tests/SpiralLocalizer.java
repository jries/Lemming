package org.lemming.tests;

import org.lemming.data.Localization;
import org.lemming.data.XYLocalization;
import org.lemming.interfaces.Source;

public class SpiralLocalizer implements Source<Localization> {

	int i = 0;
	
	int N;
	double radius, turns;
	
	public SpiralLocalizer() {
		this(100, 128, 2);
	}
	
	public SpiralLocalizer(int N, double radius, double turns){
		this.N = N;
		this.radius = radius;
		this.turns = turns;
	}
	
	@Override
	public boolean hasMoreOutputs() {
		return i<N;
	}

	@Override
	public Localization newOutput() {
		i++;
                return new XYLocalization(radius*i/N*Math.cos(2*Math.PI*i/N*turns)+128,radius*i/N*Math.sin(2*Math.PI*i/N*turns)+128);
	}

	@Override
	public void beforeRun() {
        }

	@Override
	public void afterRun() {
        }

}
