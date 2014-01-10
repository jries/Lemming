package org.lemming.interfaces;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.Store;

/**
 * A localizer is transforms Frames into Localizations.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface Localizer extends Runnable {
	
	public void setInput(Store<Frame> s);
	
	public void setOutput(Store<Localization> s);
	
}
