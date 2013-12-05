package org.lemming.interfaces;

import org.lemming.data.Store;

public interface Processor<T1, T2> extends Runnable {

	public void setInput(Store<T1> s);
	
	public void setOutput(Store<T2> s);
	
}
