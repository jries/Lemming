package org.lemming.inputs;

import org.lemming.interfaces.Source;
import org.lemming.interfaces.Store;
import org.lemming.outputs.NullStoreWarning;

/**
 * This class represents a module with a single output. This typically represents a generator of objects of type T.
 * It provides a standard implementation of the run method, which checks the output before calling newOutput
 * in a loop while hasMoreOutputs is true.
 * 
 * @author Thomas Pengo
 * @author Ronny Sczech
 *
 * @param <T> Type parameter for the kind of objects that are being generated.
 */
public abstract class SingleOutput<T> implements Source<T> {

	/**
	 * output
	 */
	public Store<T> output;

	@Override
	public final void run() {
		if (output==null)
			throw new NullStoreWarning(this.getClass().getName()); 
		
		beforeRun();
		
		while (hasMoreOutputs()) {
			output.put(newOutput());
		}
		
		afterRun();
	}
	
	/**
	 * This method is called in a loop to generate a new output. It is expected to generate a new object of type T.
	 * 
	 * @return the newly created object
	 */
	protected abstract T newOutput();
	
	@Override
	public void setOutput(Store<T> s) {
		output = s;
	}
	
	/**
	 * 
	 */
	public void beforeRun() {}
	
	/**
	 * 
	 */
	public void afterRun() {}

}
