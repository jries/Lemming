package org.lemming.interfaces;

import org.lemming.data.Store;

/**
 * A processor transforms T1s into T2s.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T1> the input types
 * @param <T2> the output types
 */
public interface Processor<T1, T2> extends Runnable {

	public void setInput(Store<T1> s);
	
	public void setOutput(Store<T2> s);
	
}
