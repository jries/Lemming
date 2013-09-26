package org.lemming.interfaces;

import org.lemming.data.Frame;
import org.lemming.data.Store;

public interface Input extends Runnable {

	public void setOutput(Store<Frame> store);
	
}
