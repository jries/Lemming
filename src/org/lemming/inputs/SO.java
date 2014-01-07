package org.lemming.inputs;

import org.lemming.data.Store;
import org.lemming.interfaces.Source;
import org.lemming.outputs.NullStoreWarning;

public abstract class SO<T> implements Source<T>, Runnable {

	Store<T> output;

	@Override
	public void run() {
		
		if (output==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		while (hasMoreOutputs()) {
			output.put(newOutput());
		}
	}
	
	public abstract T newOutput();
	
	@Override
	public void setOutput(Store<T> s) {
		output = s;
	}

}
