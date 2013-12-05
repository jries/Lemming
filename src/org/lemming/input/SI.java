package org.lemming.input;

import org.lemming.data.Store;
import org.lemming.interfaces.Well;
import org.lemming.outputs.NullStoreWarning;

public abstract class SI<T> implements Well<T>, Runnable {

	Store<T> input;

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
