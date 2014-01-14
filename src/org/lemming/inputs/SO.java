package org.lemming.inputs;

import org.lemming.data.Store;
import org.lemming.interfaces.Source;
import org.lemming.outputs.NullStoreWarning;

/**
 * This class represents a module with a single output. This typically represents a generator of objects of type T.
 * It provides a standard implementation of the run method, which checks the output before calling newOutput
 * in a loop while hasMoreOutputs is true.
 * 
 * @author Thomas Pengo
 *
 * @param <T> Type parameter for the kind of objects that are being generated.
 */
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
	
	/**
	 * This method is called in a loop to generate a new output. It is expected to generate a new object of type T.
	 * 
	 * @return the newly created object
	 */
	public abstract T newOutput();
	
	@Override
	public void setOutput(Store<T> s) {
		output = s;
	}

}
