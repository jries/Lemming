package org.lemming.interfaces;


/**
 * A processor transforms T1s into T2s.
 * 
 * @author Thomas Pengo, Joe Borbely, Ronny Sczech
 *
 * @param <T1> the input types
 * @param <T2> the output types
 */
public interface Processor<T1, T2> extends Runnable {

	/**
	 * @param s - the specified Store
	 */
	public void setInput(Store<T1> s);
	
	/**
	 * @param s - the specified Store
	 */
	public void setOutput(Store<T2> s);	
	
	/**
	 * @return Returns true if the Source has more outputs to be generated.
	 */
	public boolean hasMoreOutputs();
}
