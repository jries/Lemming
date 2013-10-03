package org.lemming.interfaces;

import org.lemming.data.Store;

public interface Source<T> extends Runnable {

	public void setOutput(Store<T> store);
	
	public boolean hasMoreOutputs();
}
