package org.lemming.interfaces;

import org.lemming.data.Localization;
import org.lemming.data.Store;

public interface Processor {

	public void setInput(Store<Localization> s);
	
	public void setOutput(Store<Localization> s);
	
}
