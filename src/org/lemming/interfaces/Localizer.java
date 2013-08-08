package org.lemming.interfaces;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.Store;

public interface Localizer {
	
	public void setInput(Store<Frame> s);
	
	public void setOutput(Store<Localization> s);
	
}
