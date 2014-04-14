package org.lemming.queue;

import org.lemming.queue.Store;
import org.lemming.interfaces.Well;

public abstract class SI<T> implements Well<T>, Runnable {

	Store<T> input;
        Well<T> well;

	@Override
	public void run() {
		well.beforeRun();
		
		if (input==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		T loc;
		while ((loc=input.get())!=null) {
			well.process(loc);
		}
		
		well.afterRun();
	}

	@Override
	public void setInput(Store<T> s) {
		input = s;
	}

}
