package org.lemming.interfaces;

import org.lemming.data.Store;

/**
 * A source creates new T into setOutput when run.  
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T>
 */
public interface Source<T> extends Runnable {

	/**
	 * Sets the output of the source to the specified Store.
	 * 
	 * @param store
	 */
	public void setOutput(Store<T> store);
	
	/**
	 * Returns true if the Source has more outputs to be generated.
	 * @return
	 */
	public boolean hasMoreOutputs();
}
