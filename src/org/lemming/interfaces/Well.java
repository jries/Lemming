package org.lemming.interfaces;

import org.lemming.data.Store;

public interface Well<T> extends Runnable {
	
	public void setInput(Store<T> s);

}
