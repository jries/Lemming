package org.lemming.interfaces;

import org.lemming.data.Store;

/**
 * A Well when run reads and processes from the associated Store.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T>
 */
public interface Well<T> extends Runnable {
	
	public void setInput(Store<T> s);

}
