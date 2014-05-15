package org.lemming.outputs;

import org.lemming.data.Store;
import org.lemming.interfaces.Well;

public abstract class SI<T> implements Well<T> {

	protected Store<T> input;			// "protected" added to be able to call it from another package (for matlab)

	@Override
	public void run() {
		beforeRun();
		
		if (input==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		T loc;
		while ((loc=nextInput())!=null) {
			process(loc);
		}
		
		afterRun();
	}
	
	public abstract void process(T element);
	
	T nextInput() {
		return input.get();
	}

	@Override
	public void setInput(Store<T> s) {
		input = s;
	}

	public void beforeRun() {};
	public void afterRun() {};
	

}
