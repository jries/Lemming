package org.lemming.outputs;

import org.lemming.data.Frame;
import org.lemming.data.Store;
import org.lemming.interfaces.Input;

public abstract class SO implements Input, Runnable {

	Store<Frame> output;

	@Override
	public void run() {
		
		if (output==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		while (hasMoreFrames()) {
			output.put(newFrame());
		}
	}
	
	public abstract boolean hasMoreFrames();
	public abstract Frame newFrame();
	
	@Override
	public void setOutput(Store<Frame> s) {
		output = s;
	}

}
