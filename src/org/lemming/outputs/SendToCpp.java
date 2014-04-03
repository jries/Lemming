package org.lemming.outputs;

public class SendToCpp extends SI<Double>{
	@Override
	public void process(Double element) {
		System.out.println(element);
		
	}
}
