package org.lemming.interfaces;

import org.lemming.data.Localization;
import org.lemming.data.Store;

public interface Output extends Runnable {
	
	public void setInput(Store<Localization> s);

}
