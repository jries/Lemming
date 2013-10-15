package org.lemming.processor;

import org.lemming.data.Store;
import org.lemming.interfaces.Processor;
import org.lemming.outputs.NullStoreWarning;

public abstract class SISO<T1,T2> implements Runnable, Processor<T1, T2> {

	protected Store<T1> input;
	protected Store<T2> output;

	@Override
	public void run() {
		
		if (input==null || output==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		T1 loc;
		while ((loc=nextInput())!=null) {
			process(loc);
		}
	}
	
	public abstract void process(T1 element);
	
	T1 nextInput() {
		return input.get();
	}

	@Override
	public void setInput(Store<T1> s) {
		input = s;
	}

	@Override
	public void setOutput(Store<T2> s) {
		output = s;
	}

}
