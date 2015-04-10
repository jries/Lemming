package org.lemming.interfaces;


/**
 * A source creates new T into setOutput when run.  
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T> - data type
 */
public interface Source<T> extends Runnable {

	/**
	 * Sets the output of the source to the specified Store.
	 * 
	 * @param store - the specified Store
	 */
	public void setOutput(Store<T> store);
	
	/**
	 * @return Returns true if the Source has more outputs to be generated.
	 */
	public boolean hasMoreOutputs();
}
