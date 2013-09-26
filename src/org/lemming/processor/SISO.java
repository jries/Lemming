package org.lemming.processor;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Processor;
import org.lemming.outputs.NullStoreWarning;

public abstract class SISO implements Runnable, Processor {

	Store<Localization> input;
	Store<Localization> output;

	@Override
	public void run() {
		
		if (input==null || output==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		Localization loc;
		while ((loc=nextInput())!=null) {
			process(loc);
		}
	}
	
	public abstract void process(Localization element);
	
	Localization nextInput() {
		return input.get();
	}

	@Override
	public void setInput(Store<Localization> s) {
		input = s;
	}

	@Override
	public void setOutput(Store<Localization> s) {
		output = s;
	}

}
