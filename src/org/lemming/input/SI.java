package org.lemming.input;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Output;
import org.lemming.outputs.NullStoreWarning;

public abstract class SI implements Output, Runnable {

	Store<Localization> input;

	@Override
	public void run() {
		
		if (input==null)
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


}
