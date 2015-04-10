package org.lemming.outputs;

import org.lemming.interfaces.Store;
import org.lemming.interfaces.Well;

/**
 * @author Ronny Sczech
 *
 * @param <T> - data type
 */
public abstract class SingleInput<T> implements Well<T> {

	protected Store<T> input;
	private volatile boolean running = true;

	@Override
	public void run() {
		beforeRun();
		
		if (input==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		while (running) {
			process(nextInput());			
		}
		
		afterRun();
	}
	
	/**
	 * @param element - element
	 */
	public abstract void process(T element);
	
	T nextInput() {
		return input.get();
	}
	
	/**
	 * 
	 */
	public void stop(){
		running = false;
	}

	@Override
	public void setInput(Store<T> s) {
		input = s;
	}

	/**
	 * 
	 */
	public void beforeRun() {};
	/**
	 * 
	 */
	public void afterRun() {};	

}
