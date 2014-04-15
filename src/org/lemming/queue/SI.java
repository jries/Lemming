package org.lemming.queue;

import org.lemming.queue.Store;
import org.lemming.interfaces.Well;

public class SI<T> implements Runnable {

	Store<T> input;
        Well<T> well;

        public SI(Store<T> input, Well<T> well) {
            this.input = input;
            this.well = well;
        }

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

}
